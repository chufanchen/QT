import argparse
import os
import pathlib
import pickle
import random
import sys
import time

import d4rl
import gym
import numpy as np
import torch
import wandb
# from torch.utils.tensorboard import SummaryWriter

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
    evaluate_episode_rtg_v1,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import GaussianBCModel, MLPBCModel
from decision_transformer.models.model import Critic
from decision_transformer.models.ql_DT import QDecisionTransformer
from decision_transformer.training.act_trainer import ActTrainer, StochasticActTrainer
from decision_transformer.training.ql_trainer import QDTTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from tqdm import tqdm, trange


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 8  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def save_checkpoint(state, name):
    filename = name
    torch.save(state, filename)


def load_model(path, model, critic=None):
    """
    Load saved model and critic state dictionaries from a checkpoint file

    Args:
        path: Path to the checkpoint file
        model: The actor model to load state into
        critic: The critic model to load state into (optional)

    Returns:
        epoch: The epoch number when the model was saved
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["actor"])
    if critic is not None and checkpoint["critic"] is not None:
        critic.load_state_dict(checkpoint["critic"])
    return checkpoint["epoch"]


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment(
    exp_prefix,
    variant,
):
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    seed = variant["seed"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    timestr = time.strftime("%y%m%d-%H%M%S")
    exp_prefix = f"{group_name}-{seed}-{timestr}"

    if not os.path.exists(os.path.join(variant["save_path"], exp_prefix)):
        pathlib.Path(args.save_path + exp_prefix).mkdir(parents=True, exist_ok=True)

    if env_name == "hopper":
        dversion = 2
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        dversion = 2
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 9000, 6000]
        scale = 1000.0
    elif env_name == "walker2d":
        dversion = 2
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [5000, 4000, 2500]
        scale = 1000.0
    elif env_name == "reacher2d":
        # from decision_transformer.envs.reacher_2d import Reacher2dEnv
        # env = Reacher2dEnv()
        env = gym.make("Reacher-v4")
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.0
        dversion = 2
    elif env_name == "pen":
        dversion = 1
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
    elif env_name == "hammer":
        dversion = 1
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.0
    elif env_name == "door":
        dversion = 1
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 100.0
    elif env_name == "relocate":
        dversion = 1
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [3000, 1000]
        scale = 1000.0
        dversion = 1
    elif env_name == "kitchen":
        dversion = 0
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.0
    elif env_name == "maze2d":
        if "open" in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [300, 200, 150, 100, 50, 20]
        scale = 10.0
    elif env_name == "antmaze":
        dversion = 0
        gym_name = f"{env_name}-{dataset}-v{dversion}"
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
        scale = 1.0
    else:
        raise NotImplementedError

    if variant["scale"] is not None:
        scale = variant["scale"]

    variant["max_ep_len"] = max_ep_len

    if model_type == "bc":
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    variant["env_targets"] = env_targets

    variant["scale"] = scale
    if variant["test_scale"] is None:
        variant["test_scale"] = scale

    env.seed(variant["seed"])
    set_seed(variant["seed"])

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f"D4RL/{env_name}-{dataset}-v{dversion}.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get("mode", "normal")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    num_eval_episodes = variant["num_eval_episodes"]
    pct_traj = variant.get("pct_traj", 1.0)

    if variant["create_pct_traj_and_exit"]:
        # Sort trajectories by return (lowest to highest)
        sorted_inds = np.argsort(returns)

        # Calculate how many trajectories to keep based on pct_traj
        num_timesteps_to_keep = max(int(pct_traj * num_timesteps), 1)
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2

        # Calculate which trajectories to keep
        while (
            ind >= 0
            and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps_to_keep
        ):
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1

        # Get indices of trajectories to keep (highest return trajectories)
        keep_inds = sorted_inds[-num_trajectories:]
        
        # Create a set of indices to keep for fast lookup
        keep_inds_set = set(keep_inds)
        
        # Create new dataset with only the top trajectories
        filtered_trajectories = [trajectories[i] for i in keep_inds]
        mode = variant.get("mode", "normal")
        states_, traj_lens_, returns_= [], [], []
        for path in filtered_trajectories:
            if mode == "delayed":  # delayed: all rewards moved to end of trajectory
                path["rewards"][-1] = path["rewards"].sum()
                path["rewards"][:-1] = 0.0
            states_.append(path["observations"])
            traj_lens_.append(len(path["observations"]))
            returns_.append(path["rewards"].sum())
        traj_lens_, returns_ = np.array(traj_lens_), np.array(returns_)

        # used for input normalization
        states_ = np.concatenate(states_, axis=0)
        num_timesteps_ = sum(traj_lens_)

        print("=" * 50)
        print(f"Filtered {int(pct_traj*100)}% : {env_name} {dataset}")
        print(f"{len(traj_lens_)} trajectories, {num_timesteps_} timesteps found")
        print(f"Average return: {np.mean(returns_):.2f}, std: {np.std(returns_):.2f}")
        print(f"Max return: {np.max(returns_):.2f}, min: {np.min(returns_):.2f}")
        print("=" * 50)
        # Save the filtered dataset
        filtered_dataset_path = (
            f"D4RL/{env_name}-{dataset}-v{dversion}_filtered_{int(pct_traj*100)}%.pkl"
        )
        with open(filtered_dataset_path, "wb") as f:
            pickle.dump(filtered_trajectories, f)
            
        # Create augmented version of the original dataset
        augmented_trajectories = []
        for i, traj in enumerate(trajectories):
            # Create a copy of the trajectory to avoid modifying the original
            augmented_traj = traj.copy()
            
            # Add mask indicating if this trajectory is in the top percentile
            is_top_traj = i in keep_inds_set
            augmented_traj["pct_traj_mask"] = is_top_traj
            
            augmented_trajectories.append(augmented_traj)
            
        # Save the augmented dataset
        augmented_dataset_path = (
            f"D4RL/{env_name}-{dataset}-v{dversion}_augmented_{int(pct_traj*100)}%.pkl"
        )
        with open(augmented_dataset_path, "wb") as f:
            pickle.dump(augmented_trajectories, f)

        print(
            f"Saved filtered dataset with {num_trajectories}/{len(trajectories)} trajectories"
        )
        print(f"Original timesteps: {num_timesteps}, Filtered timesteps: {timesteps}")
        print(f"Filtered dataset saved to: {filtered_dataset_path}")
        print(f"Augmented dataset saved to: {augmented_dataset_path}")

        # Exit without training
        sys.exit(0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if "hopper-medium" in gym_name:
                si = random.randint(0, traj["rewards"].shape[0] - K - 1)
            else:
                si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            target_a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1, 1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff

            if variant["reward_tune"] == "cql_antmaze":
                traj_rewards = (traj["rewards"] - 0.5) * 4.0
            else:
                traj_rewards = traj["rewards"]
            r.append(traj_rewards[si : si + max_len].reshape(1, -1, 1))
            rtg.append(
                discount_cumsum(traj_rewards[si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1
            )
            d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, target_a, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model, critic=None):
            returns, lengths = [], []
            for _ in trange(num_eval_episodes):
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_rtg_v1(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=variant["test_scale"],
                            target_return=target_rew / variant["test_scale"],
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    elif model_type == "qdt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            critic,
                            max_ep_len=max_ep_len,
                            scale=variant["test_scale"],
                            target_return=[
                                t / variant["test_scale"] for t in target_rew
                            ],
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:  # bc
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
                f"target_{target_rew}_normalized_score": env.get_normalized_score(
                    np.mean(returns)
                ),
            }

        return fn

    if model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    elif model_type == "qdt":
        model = QDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            scale=scale,
            sar=variant["sar"],
            rtg_no_q=variant["rtg_no_q"],
            infer_no_q=variant["infer_no_q"],
            stochastic_policy=variant["stochastic_policy"],
        )
    elif model_type == "bc":
        if variant["stochastic_policy"]:
            model = GaussianBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
            )
        else:
            model = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
                dropout=0.0,  # TODO: REMOVE after debug
            )
    else:
        raise NotImplementedError

    model = model.to(device=device)
    prior = None
    if model_type == "qdt":
        critic = Critic(state_dim, act_dim, hidden_dim=variant["embed_dim"])
        critic = critic.to(device=device)
        if variant["policy_penalty"] or variant["value_penalty"]:
            prior = GaussianBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
            )
            load_model(variant["behavior_ckpt_file"], prior)
            prior = prior.to(device=device)
    else:
        warmup_steps = variant["warmup_steps"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
        )

    if model_type == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == "qdt":
        trainer = QDTTrainer(
            model=model,
            critic=critic,
            batch_size=batch_size,
            tau=variant["tau"],
            discount=variant["discount"],
            get_batch=get_batch,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(env_targets)],
            max_q_backup=variant["max_q_backup"],
            alpha=variant["alpha"],
            eta=variant["eta"],
            eta2=variant["eta2"],
            ema_decay=0.995,
            step_start_ema=1000,
            update_ema_every=5,
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            lr_decay=variant["lr_decay"],
            lr_maxt=variant["max_iters"],
            lr_min=variant["lr_min"],
            grad_norm=variant["grad_norm"],
            scale=scale,
            k_rewards=variant["k_rewards"],
            use_discount=variant["use_discount"],
            prior=prior,
            policy_penalty=variant["policy_penalty"],
            value_penalty=variant["value_penalty"],
            action_spec=env.action_space,
        )
    elif model_type == "bc":
        if variant["stochastic_policy"]:
            trainer = StochasticActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
                    (a_hat - a) ** 2
                ),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        else:
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
                    (a_hat - a) ** 2
                ),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        wandb.watch(model)

    best_ret = -10000
    best_nor_ret = -1000
    best_iter = -1
    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        if log_to_wandb:
            wandb.log(outputs)
        if model_type == "qdt":
            trainer.scale_up_eta(variant["lambda"])
        ret = outputs["Best_return_mean"]
        nor_ret = outputs["Best_normalized_score"]
        if ret > best_ret:
            state = {
                "epoch": iter + 1,
                "actor": trainer.model.state_dict(),
                "critic": trainer.critic_target.state_dict()
                if model_type == "qdt"
                else None,
            }
            save_checkpoint(
                state,
                os.path.join(
                    variant["save_path"], exp_prefix, "epoch_{}.pth".format(iter + 1)
                ),
            )
            best_ret = ret
            best_nor_ret = nor_ret
            best_iter = iter + 1
        print(
            f"Current best return mean is {best_ret}, normalized score is {best_nor_ret * 100}, Iteration {best_iter}"
        )

        if variant["early_stop"] and iter >= variant["early_epoch"]:
            break

    wandb.log(
        {
            "final_best_return_mean": best_ret,
            "final_best_normalized_score": best_nor_ret * 100,
        }
    )
    print(f"The final best return mean is {best_ret}")
    print(f"The final best normalized return is {best_nor_ret * 100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="gym-experiment")
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="./save/")

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert

    parser.add_argument("--model_type", type=str, default="dt")
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument(
        "--create_pct_traj_and_exit", action="store_true", default=False
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=0.0)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--num_steps_per_iter", type=int, default=1000)

    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--eta", default=1.0, type=float)
    parser.add_argument("--eta2", default=1.0, type=float)
    parser.add_argument("--lambda", default=1.0, type=float)
    parser.add_argument("--max_q_backup", action="store_true", default=False)
    parser.add_argument("--lr_decay", action="store_true", default=False)
    parser.add_argument("--grad_norm", default=2.0, type=float)
    parser.add_argument("--early_stop", action="store_true", default=False)
    parser.add_argument("--early_epoch", type=int, default=100)
    parser.add_argument("--k_rewards", action="store_true", default=False)
    parser.add_argument("--use_discount", action="store_true", default=False)
    parser.add_argument("--sar", action="store_true", default=False)
    parser.add_argument("--reward_tune", default="no", type=str)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--test_scale", type=float, default=None)
    parser.add_argument("--rtg_no_q", action="store_true", default=False)
    parser.add_argument("--infer_no_q", action="store_true", default=False)

    parser.add_argument("--stochastic_policy", action="store_true", default=False)
    parser.add_argument("--behavior_ckpt_file", type=str, default=None)
    parser.add_argument("--policy_penalty", action="store_true", default=False)
    parser.add_argument("--value_penalty", action="store_true", default=False)

    args = parser.parse_args()

    experiment(args.exp_name, variant=vars(args))
