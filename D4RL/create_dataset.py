import gym
import numpy as np

import collections
import pickle

import d4rl


datasets = []

# names = [
#     "halfcheetah-medium-expert-v2",
#     "halfcheetah-medium-replay-v2",
#     "halfcheetah-medium-v2",
#     "hopper-medium-expert-v2",
#     "hopper-medium-replay-v2",
#     "hopper-medium-v2",
#     "walker2d-medium-expert-v2",
#     "walker2d-medium-replay-v2",
#     "walker2d-medium-v2",
# ]

names = ["kitchen-complete-v0", "kitchen-partial-v0"]

# names = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"]


for env_name in names:
    name = env_name
    env = gym.make(name)
    print(env.observation_space)
    dataset = env.get_dataset()
    print(dataset.keys())

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == 1000 - 1
        for k in ["observations", "actions", "rewards", "terminals"]:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )

    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(paths, f)
