# Create mixed datasets from D4RL datasets
# E.g. halfcheetah-mixed-v2 is a mix of halfcheetah-medium-replay-v2, halfcheetah-medium-v2 and halfcheetah-medium-expert-v2
import d4rl
import gym
import numpy as np
import pickle
import collections

def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_mixed_dataset(dataset_paths, output_path, mix_ratios=None, total_trajectories=None):
    """
    Create a mixed dataset from multiple D4RL datasets
    Args:
        dataset_paths: list of paths to dataset pickle files
        output_path: path to save the mixed dataset
        mix_ratios: optional list of mixing ratios (will be normalized to sum to 1)
        total_trajectories: optional total number of trajectories to include
    """
    # Load all datasets
    datasets = [load_dataset(path) for path in dataset_paths]
    
    # If mix_ratios not provided, use equal ratios
    if mix_ratios is None:
        mix_ratios = [1.0 / len(datasets)] * len(datasets)
    
    # Normalize mix_ratios to sum to 1
    mix_ratios = np.array(mix_ratios)
    mix_ratios = mix_ratios / mix_ratios.sum()
    
    # Extract trajectories from each dataset
    all_trajectories = []
    for dataset in datasets:
        # Get episode start indices
        episode_starts = [0] + [i + 1 for i in range(len(dataset['terminals'])) if dataset['terminals'][i]]
        episode_ends = [i + 1 for i in range(len(dataset['terminals'])) if dataset['terminals'][i]] + [len(dataset['terminals'])]
        
        # Extract individual trajectories
        trajectories = []
        for start, end in zip(episode_starts, episode_ends):
            trajectory = {
                'observations': dataset['observations'][start:end],
                'actions': dataset['actions'][start:end],
                'rewards': dataset['rewards'][start:end],
                'terminals': dataset['terminals'][start:end],
                'timeouts': dataset['timeouts'][start:end] if 'timeouts' in dataset else np.zeros_like(dataset['terminals'][start:end])
            }
            trajectories.append(trajectory)
        all_trajectories.append(trajectories)
    
    # Calculate number of trajectories to sample from each dataset
    if total_trajectories is None:
        total_trajectories = sum(len(trajs) for trajs in all_trajectories)
    
    n_trajectories = [int(ratio * total_trajectories) for ratio in mix_ratios]
    # Adjust for rounding errors
    n_trajectories[-1] = total_trajectories - sum(n_trajectories[:-1])
    
    # Sample trajectories according to mix_ratios
    mixed_trajectories = []
    for trajs, n in zip(all_trajectories, n_trajectories):
        print(f"Sampling {n} trajectories...")
        indices = np.random.choice(len(trajs), size=n, replace=True)
        mixed_trajectories.extend([trajs[i] for i in indices])
    
    # Combine trajectories into a single dataset
    combined_dataset = {
        'observations': np.concatenate([t['observations'] for t in mixed_trajectories]),
        'actions': np.concatenate([t['actions'] for t in mixed_trajectories]),
        'rewards': np.concatenate([t['rewards'] for t in mixed_trajectories]),
        'terminals': np.concatenate([t['terminals'] for t in mixed_trajectories]),
        'timeouts': np.concatenate([t['timeouts'] for t in mixed_trajectories])
    }
    
    # Save the mixed dataset
    with open(output_path, 'wb') as f:
        pickle.dump(combined_dataset, f)


if __name__ == "__main__":
    # Example usage for halfcheetah mixed dataset
    # datasets = [
    #     "halfcheetah-medium-expert-v2.pkl",
    #     "halfcheetah-medium-replay-v2.pkl",
    #     "halfcheetah-medium-v2.pkl"
    # ]
    
    datasets = [
        "maze2d_expert.pkl",
        "maze2d_medium_expert.pkl",
        "maze2d_random.pkl"
    ]
    
    # datasets = [
    #     "hopper-medium-expert-v2.pkl",
    #     "hopper-medium-replay-v2.pkl",
    #     "hopper-medium-v2.pkl"
    # ]
    
    # datasets = [
    #     "walker2d-medium-expert-v2.pkl",
    #     "walker2d-medium-replay-v2.pkl",
    #     "walker2d-medium-v2.pkl"
    # ]
    
    # Create with equal mixing ratios
    create_mixed_dataset(
        dataset_paths=datasets,
        output_path="maze2d-medium-mixed-v1.pkl",
        mix_ratios=[1/7, 5/7, 1/7],
        total_trajectories=600
    )
    
    create_mixed_dataset(
        dataset_paths=datasets,
        output_path="maze2d-medium-mixed-v2.pkl",
        mix_ratios=[1/7, 5/7, 1/7],
        total_trajectories=300
    )
    
    create_mixed_dataset(
        dataset_paths=datasets,
        output_path="maze2d-medium-mixed-v3.pkl",
        mix_ratios=[5/14, 2/7, 5/14],
        total_trajectories=600
    )
    
    create_mixed_dataset(
        dataset_paths=datasets,
        output_path="maze2d-medium-mixed-v4.pkl",
        mix_ratios=[5/14, 2/7, 5/14],
        total_trajectories=300
    )
    
    # create_mixed_dataset(
    #     dataset_paths=datasets,
    #     output_path="hopper-mixed-v2.pkl"
    # )
    
    # create_mixed_dataset(
    #     dataset_paths=datasets,
    #     output_path="hopper-mixed-v2.pkl"
    # )
    
    # create_mixed_dataset(
    #     dataset_paths=datasets,
    #     output_path="walker2d-mixed-v2.pkl"
    # )
    
    # Or with custom mixing ratios (e.g., 0.4, 0.3, 0.3)
    # create_mixed_dataset(
    #     dataset_paths=datasets,
    #     output_path="halfcheetah-mixed-custom-v2.pkl",
    #     mix_ratios=[0.4, 0.3, 0.3]
    # )


