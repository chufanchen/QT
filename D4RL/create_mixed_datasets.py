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

def create_mixed_dataset(dataset_paths, output_path, mix_ratios=None):
    """
    Create a mixed dataset from multiple D4RL datasets
    Args:
        dataset_paths: list of paths to dataset pickle files
        output_path: path to save the mixed dataset
        mix_ratios: optional list of mixing ratios (will be normalized to sum to 1)
    """
    # Load all datasets
    datasets = [load_dataset(path) for path in dataset_paths]
    
    # If no mix_ratios provided, use equal ratios
    if mix_ratios is None:
        mix_ratios = [1/len(datasets)] * len(datasets)
    else:
        # Normalize ratios to sum to 1
        mix_ratios = np.array(mix_ratios) / np.sum(mix_ratios)
    
    # Calculate number of trajectories to take from each dataset
    dataset_sizes = [len(dataset) for dataset in datasets]
    total_size = sum(dataset_sizes)
    num_trajectories = [min(int(ratio * total_size), size) 
                       for ratio, size in zip(mix_ratios, dataset_sizes)]
    
    # Randomly sample trajectories from each dataset
    mixed_paths = []
    for dataset, n_traj in zip(datasets, num_trajectories):
        indices = np.random.choice(len(dataset), size=n_traj, replace=False)
        mixed_paths.extend([dataset[i] for i in indices])
    
    # Shuffle the mixed dataset
    np.random.shuffle(mixed_paths)
    
    # Print statistics
    returns = np.array([np.sum(p["rewards"]) for p in mixed_paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in mixed_paths])
    print(f"Mixed dataset statistics:")
    print(f"Number of trajectories: {len(mixed_paths)}")
    print(f"Number of samples: {num_samples}")
    print(f"Returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}")
    print(f"Returns: min = {np.min(returns):.2f}, max = {np.max(returns):.2f}")
    
    # Save mixed dataset
    with open(output_path, 'wb') as f:
        pickle.dump(mixed_paths, f)

if __name__ == "__main__":
    # Example usage for halfcheetah mixed dataset
    datasets = [
        "halfcheetah-medium-expert-v2.pkl",
        "halfcheetah-medium-replay-v2.pkl",
        "halfcheetah-medium-v2.pkl"
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
        output_path="halfcheetah-mixed-v2.pkl"
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


