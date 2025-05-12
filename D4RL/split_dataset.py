import d4rl
import gym
import numpy as np
import pickle

def split_dataset_by_return(env_name='maze2d-umaze-v1', expert_threshold=0.3, medium_threshold=0.3):
    """
    Split a D4RL dataset into expert, medium-expert, and random based on returns
    Args:
        env_name: Name of the D4RL environment
        expert_threshold: Percentile threshold for expert data (top 30% by default)
        medium_threshold: Percentile threshold for medium data (bottom 30% is random)
    """
    # Load the environment and dataset
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    # Calculate returns for each trajectory
    returns = []
    trajectory_data = []
    
    current_trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'timeouts': []
    }
    
    total_return = 0
    
    # Group the data into trajectories
    for i in range(len(dataset['observations'])):
        current_trajectory['observations'].append(dataset['observations'][i])
        current_trajectory['actions'].append(dataset['actions'][i])
        current_trajectory['rewards'].append(dataset['rewards'][i])
        current_trajectory['terminals'].append(dataset['terminals'][i])
        current_trajectory['timeouts'].append(dataset['timeouts'][i])
        
        total_return += dataset['rewards'][i]
        
        if dataset['terminals'][i] or dataset['timeouts'][i]:
            returns.append(total_return)
            # Convert lists to numpy arrays
            trajectory_data.append({
                'observations': np.array(current_trajectory['observations']),
                'actions': np.array(current_trajectory['actions']),
                'rewards': np.array(current_trajectory['rewards']),
                'terminals': np.array(current_trajectory['terminals']),
                'timeouts': np.array(current_trajectory['timeouts'])
            })
            
            # Reset for next trajectory
            current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'terminals': [],
                'timeouts': []
            }
            total_return = 0
    
    # Calculate thresholds using argsort
    returns = np.array(returns)
    sorted_indices = np.argsort(returns)
    num_trajectories = len(returns)
    
    # Calculate split indices
    num_random = int(num_trajectories * medium_threshold)  # bottom 30%
    num_expert = int(num_trajectories * expert_threshold)  # top 30%
    
    # Split trajectories
    expert_data = [trajectory_data[i] for i in sorted_indices[-num_expert:]]  # top 30%
    random_data = [trajectory_data[i] for i in sorted_indices[:num_random]]   # bottom 30%
    medium_data = [trajectory_data[i] for i in sorted_indices[num_random:-num_expert]]  # middle 40%
    
    print(f"Return statistics:")
    print(f"Min return: {np.min(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Split sizes - Expert: {len(expert_data)}, Medium: {len(medium_data)}, Random: {len(random_data)}")
    
    # Combine trajectories into datasets
    def combine_trajectories(trajectories):
        if not trajectories:
            return None
        
        # Instead of concatenating, just return the list of trajectory dictionaries
        paths = []
        for traj in trajectories:
            paths.append({
                'observations': traj['observations'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'terminals': traj['terminals'],
                'timeouts': traj['timeouts']
            })
        return paths
    
    expert_dataset = combine_trajectories(expert_data)
    medium_dataset = combine_trajectories(medium_data)
    random_dataset = combine_trajectories(random_data)
    
    # Save datasets
    with open('maze2d_expert.pkl', 'wb') as f:
        pickle.dump(expert_dataset, f)
    with open('maze2d_medium_expert.pkl', 'wb') as f:
        pickle.dump(medium_dataset, f)
    with open('maze2d_random.pkl', 'wb') as f:
        pickle.dump(random_dataset, f)
    
    print(f"Expert trajectories: {len(expert_data)}")
    print(f"Medium trajectories: {len(medium_data)}")
    print(f"Random trajectories: {len(random_data)}")

if __name__ == "__main__":
    split_dataset_by_return()