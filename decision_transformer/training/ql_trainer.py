import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange

from decision_transformer.models.divergences import clip_by_eps, KL, MMD, FDivergence, W
from decision_transformer.training.trainer import Trainer


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class QDTTrainer(Trainer):
    def __init__(
        self,
        model,
        critic,
        batch_size,
        tau,
        discount,
        get_batch,
        loss_fn,
        eval_fns=None,
        max_q_backup=False,
        alpha=1.0,
        eta=1.0,
        eta2=1.0,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        weight_decay=1e-4,
        lr_decay=False,
        lr_maxt=100000,
        lr_min=0.0,
        grad_norm=1.0,
        scale=1.0,
        k_rewards=True,
        use_discount=True,
        prior=None,
        divergence_name="kl",
        n_div_samples=4,
        action_spec=None,
        policy_penalty=True,
        value_penalty=False,
    ):
        self.actor = model
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min
            )

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.alpha = alpha
        self.eta = eta
        self.eta2 = eta2
        self.lr_decay = lr_decay
        self.scale = scale
        self.k_rewards = k_rewards
        self.use_discount = use_discount
        self.n_div_samples = n_div_samples
        self.prior = prior
        self.policy_penalty = policy_penalty
        self.value_penalty = value_penalty
        self.divergence_name = divergence_name
        if self.divergence_name == "kl":
            self.divergence = KL
        elif self.divergence_name == "mmd":
            self.divergence = MMD
        elif self.divergence_name == "f":
            self.divergence = FDivergence
        elif self.divergence_name == "w":
            self.divergence = W
        else:
            raise ValueError("Divergence not implemented")
        self.action_spec = action_spec
        self.start_time = time.time()
        self.step = 0

    def step_ema(self):
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)

    def action_loss_fn(
        self,
        a_hat_dist,
        a,
        attention_mask,
        entropy_reg,
    ):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + entropy_reg * entropy)
        # if torch.isnan(loss):
        #     print("NAN detected in action loss")
        #     import pdb

        #     pdb.set_trace()
        return (
            loss,
            -log_likelihood,
            entropy,
        )

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
        bc_losses = []
        ql_losses = []
        actor_losses = []
        critic_losses = []

        for _ in trange(num_steps):
            loss_metric = self.train_step()
            bc_losses.append(loss_metric["bc_loss"])
            ql_losses.append(loss_metric["ql_loss"])
            actor_losses.append(loss_metric["actor_loss"])
            critic_losses.append(loss_metric["critic_loss"])

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        logs["time/training"] = time.time() - train_start

        eval_start = time.time()

        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start
        logs["training/bc_loss_mean"] = np.mean(bc_losses)
        logs["training/bc_loss_std"] = np.std(bc_losses)
        logs["training/ql_loss_mean"] = np.mean(ql_losses)
        logs["training/ql_loss_std"] = np.std(ql_losses)
        logs["training/actor_loss_mean"] = np.mean(actor_losses)
        logs["training/actor_loss_std"] = np.std(actor_losses)
        logs["training/critic_loss_mean"] = np.mean(critic_losses)
        logs["training/critic_loss_std"] = np.std(critic_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            best_ret = -10000
            best_nor_ret = -10000
            for k, v in logs.items():
                if "return_mean" in k:
                    best_ret = max(best_ret, float(v))
                if "normalized_score" in k:
                    best_nor_ret = max(best_nor_ret, float(v))
                print(k, float(v))
            print(
                "Current actor learning rate",
                self.actor_optimizer.param_groups[0]["lr"],
            )
            print(
                "Current critic learning rate",
                self.critic_optimizer.param_groups[0]["lr"],
            )

        logs["Best_return_mean"] = best_ret
        logs["Best_normalized_score"] = best_nor_ret
        return logs

    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_

    def train_step(self):
        """
        Train the model for one step
        states: (batch_size, max_len, state_dim)
        """
        (
            states,
            actions,
            rewards,
            action_target,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device

        """Q Training"""
        current_q1, current_q2 = self.critic.forward(states, actions)

        T = current_q1.shape[1]
        repeat_num = 10

        if self.max_q_backup:
            states_rpt = torch.repeat_interleave(states, repeats=repeat_num, dim=0)
            actions_rpt = torch.repeat_interleave(actions, repeats=repeat_num, dim=0)
            rewards_rpt = torch.repeat_interleave(rewards, repeats=repeat_num, dim=0)
            noise = torch.zeros(1, 1, 1)
            noise = (
                torch.cat([noise, torch.randn(repeat_num - 1, 1, 1)], dim=0)
                .repeat(batch_size, 1, 1)
                .to(device)
            )  # keep rtg logic
            rtg_rpt = torch.repeat_interleave(rtg, repeats=repeat_num, dim=0)
            rtg_rpt[:, -2:-1] = rtg_rpt[:, -2:-1] + noise * 0.1
            timesteps_rpt = torch.repeat_interleave(
                timesteps, repeats=repeat_num, dim=0
            )
            attention_mask_rpt = torch.repeat_interleave(
                attention_mask, repeats=repeat_num, dim=0
            )
            _, next_action, _ = self.ema_model(
                states_rpt,
                actions_rpt,
                rewards_rpt,
                None,
                rtg_rpt[:, :-1],
                timesteps_rpt,
                attention_mask=attention_mask_rpt,
            )
        else:
            _, next_action, _ = self.ema_model(
                states,
                actions,
                rewards,
                action_target,
                rtg[:, :-1],
                timesteps,
                attention_mask=attention_mask,
            )

        if self.k_rewards:
            if self.max_q_backup:
                critic_next_states = states_rpt[:, -1]
                next_action = (
                    next_action.sample()[:, -1]
                    if self.actor.stochastic_policy
                    else next_action[:, -1]
                )
                target_q1, target_q2 = self.critic_target(
                    critic_next_states, next_action
                )
                target_q1 = target_q1.view(batch_size, repeat_num).max(
                    dim=1, keepdim=True
                )[0]
                target_q2 = target_q2.view(batch_size, repeat_num).max(
                    dim=1, keepdim=True
                )[0]
            else:
                critic_next_states = states[:, -1]
                next_action = (
                    next_action.sample()[:, -1]
                    if self.actor.stochastic_policy
                    else next_action[:, -1]
                )
                target_q1, target_q2 = self.critic_target(
                    critic_next_states, next_action
                )
            target_q = torch.min(target_q1, target_q2)  # [B, 1]

            not_done = 1 - dones[:, -1]  # [B, 1]
            if self.use_discount:
                rewards[:, -1] = 0.0
                mask_ = attention_mask.sum(dim=1).detach().cpu()  # [B]
                discount = [i - 1 - torch.arange(i) for i in mask_]
                discount = torch.stack(
                    [torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount],
                    dim=0,
                )  # [B, T]
                discount = (
                    (self.discount**discount).unsqueeze(-1).to(device)
                )  # [B, T, 1]
                k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(
                    dims=[1]
                )  # [B, T, 1]

                discount = [torch.arange(i) for i in mask_]  #
                discount = torch.stack(
                    [torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount],
                    dim=0,
                )
                discount = (self.discount**discount).unsqueeze(-1).to(device)
                k_rewards = k_rewards / discount

                discount = [i - 1 - torch.arange(i) for i in mask_]  # [B]
                discount = torch.stack(
                    [torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount],
                    dim=0,
                )
                discount = (self.discount**discount).to(device)  # [B, T]
                target_q = (
                    k_rewards + (not_done * discount * target_q).unsqueeze(-1)
                ).detach()  # [B, T, 1]

            else:
                k_rewards = (rtg[:, :-1] - rtg[:, -2:-1]) * self.scale  # [B, T, 1]
                target_q = (
                    k_rewards + (not_done * target_q).unsqueeze(-1)
                ).detach()  # [B, T, 1]
        else:
            if self.max_q_backup:
                target_q1, target_q2 = self.critic_target(
                    states_rpt,
                    next_action.sample()
                    if self.actor.stochastic_policy
                    else next_action,
                )  # [B*repeat, T, 1]
                target_q1 = target_q1.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
                target_q2 = target_q2.view(batch_size, repeat_num, T, 1).max(dim=1)[0]
            else:
                target_q1, target_q2 = self.critic_target(
                    states,
                    next_action.sample()
                    if self.actor.stochastic_policy
                    else next_action,
                )  # [B, T, 1]
            target_q = torch.min(target_q1, target_q2)  # [B, T, 1]
            target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
            target_q = torch.cat(
                [target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1
            )

        critic_loss = F.mse_loss(
            current_q1[:, :-1][attention_mask[:, :-1] > 0],
            target_q[:, :-1][attention_mask[:, :-1] > 0],
        ) + F.mse_loss(
            current_q2[:, :-1][attention_mask[:, :-1] > 0],
            target_q[:, :-1][attention_mask[:, :-1] > 0],
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
            )
        self.critic_optimizer.step()
        # torch.autograd.set_detect_anomaly(True, check_nan=True)
        """Policy Training"""
        state_preds, action_preds, reward_preds = self.actor.forward(
            states,
            actions,
            rewards,
            action_target,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )
        action_target_ = action_target.reshape(-1, action_dim)[
            attention_mask.reshape(-1) > 0
        ]
        if self.actor.stochastic_policy:
            action_dist = action_preds
            # the return action is a SquashNormal distribution
            # action_preds_ = action_preds.rsample() # for reparameterization trick (mean + std * N(0,1))
            # action_preds_ = action_preds.sample()
            # NLL in Online DT
            action_loss, nll, entropy = self.action_loss_fn(
                action_dist,  # a_hat_dist
                clip_by_eps(action_target, self.action_spec, 1e-6),  # (batch_size, context_len, action_dim)
                attention_mask,
                self.actor.temperature().detach(),  # no gradient taken here
            )
            action_preds_ = action_dist.sample().reshape(-1, action_dim)[
                attention_mask.reshape(-1) > 0
            ]
        else:
            action_preds_ = action_preds.reshape(-1, action_dim)[
                attention_mask.reshape(-1) > 0
            ]
            action_loss = F.mse_loss(action_preds_, action_target_)
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[
            attention_mask[:, :-1] > 0
        ].mean()
        if reward_preds is not None:
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = (
                rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            )
            rewards_loss = F.mse_loss(reward_preds, reward_target)
        else:
            rewards_loss = 0
        bc_loss = action_loss + states_loss + rewards_loss

        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        # q1_new_action, q2_new_action = self.critic(state_target, action_preds_)
        if np.random.uniform() > 0.5:
            q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()

        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        if self.divergence is not None:
            # action_dist = self.actor.forward(
            #     states,
            #     actions,
            #     rewards,
            #     action_target,
            #     rtg[:, :-1],
            #     timesteps,
            #     attention_mask=attention_mask,
            # )
            _, prior_dist, _ = self.prior.forward(states, _, _)

            apn = action_dist.rsample((self.n_div_samples,))
            apn_logp = action_dist.log_prob(apn)
            # abn = prior_dist.sample_n(self.n_div_samples)
            # abn_logb = prior_dist.log_prob(apn)
            apn_logb = prior_dist.log_prob(
                clip_by_eps(apn[:, :, -1], self.action_spec, 1e-6)
            )
            # abn_logp = action_dist.log_prob(clip_by_eps(abn, self.action_spec, 1e-6))

            # # override sample
            # div_estimate = self.divergence.primal_estimate(
            #     action_dist,
            #     prior_dist,
            #     self.n_div_samples,
            #     self.action_spec,
            # )
            div_estimate = torch.mean(apn_logp[:, :, -1] - apn_logb)
            # print(div_estimate)
            actor_loss -= self.alpha * div_estimate

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
            )
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.step += 1

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds_ - action_target_) ** 2).detach().cpu().item()
            )

        loss_metric = dict()
        loss_metric["bc_loss"] = bc_loss.item()
        loss_metric["ql_loss"] = q_loss.item()
        loss_metric["critic_loss"] = critic_loss.item()
        loss_metric["actor_loss"] = actor_loss.item()

        return loss_metric
