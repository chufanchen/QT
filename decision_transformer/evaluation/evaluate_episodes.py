import numpy as np
import torch


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    with torch.no_grad():
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
        sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):
            
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
            action = model.get_action(
                (states[-1,:].to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return
            )            
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length

def evaluate_episode_rtg_v1(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        critic,
        max_ep_len=2000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    if len(ep_return) > 1:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).unsqueeze(-1)
    else:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]

    sim_states = []

    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=1).to(dtype=torch.float32),
                critic,
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

            action = action.detach().cpu().numpy().flatten()

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[:,-1:] - (reward/scale)
            else:
                pred_return = target_return[:,-1:]
            target_return = torch.cat(
                [target_return, pred_return], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

    return episode_return, episode_length

def sample_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        critic,
        max_ep_len=2000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    res_observations = []
    res_actions = []
    res_rewards = []
    res_terminals = []
    res_goals = []
    res_qpos = []
    res_qvel = []
            
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = [torch.zeros((1, act_dim), device=device, dtype=torch.float32)]
    ep_return = target_return
    if len(ep_return) > 1:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).unsqueeze(-1)
    else:
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]

    sim_states = []
    done = False
    episode_return, episode_length = 0, 0
    with torch.no_grad():
        for t in range(max_ep_len):
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=1).to(dtype=torch.float32),
                critic,
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

            action = action.detach().cpu().numpy().flatten()
            res_observations.append(state)
            res_actions.append(action)
            res_rewards.append(0.0)
            res_terminals.append(done)
            res_goals.append(env._target)
            res_qpos.append(env.sim.data.qpos.ravel().copy())
            res_qvel.append(env.sim.data.qvel.ravel().copy())
            
            if done:
                break

            state, reward, done, _ = env.step(action)
            
            actions.insert(-1, torch.from_numpy(action).reshape(1, act_dim).to(device))
            rewards.insert(-1, torch.tensor(reward).reshape(1).unsqueeze(0).to(device))

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            if mode != 'delayed':
                pred_return = target_return[:,-1:] - (reward/scale)
            else:
                pred_return = target_return[:,-1:]
            target_return = torch.cat(
                [target_return, pred_return], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1 

    return res_observations, res_actions, res_rewards, res_terminals, res_goals, res_qpos, res_qvel