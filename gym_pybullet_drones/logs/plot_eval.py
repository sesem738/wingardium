import numpy as np
import matplotlib.pyplot as plt

def read_eval_logs(file_path):
    # Load the .npz file
    data = np.load(file_path)
    
    # Extract data arrays
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    # Calculate mean rewards and episode lengths
    mean_rewards = results.mean(axis=1)
    mean_ep_lengths = ep_lengths.mean(axis=1)
    
    return timesteps, mean_rewards, mean_ep_lengths

def plot_eval_data(timesteps, mean_rewards, mean_ep_lengths, output_file):
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot mean rewards over time
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, mean_rewards, label='Mean Reward')
    plt.title('Mean Reward over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.legend()
    
    # Plot mean episode lengths over time
    plt.subplot(1, 2, 2)
    plt.plot(timesteps, mean_ep_lengths, label='Mean Episode Length', color='orange')
    plt.title('Mean Episode Length over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Length')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot as an image file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Path to the .npz file
    file_path = "evaluations.npz"
    
    # Output image file path
    output_file = "evaluation_plots.png"
    
    # Read evaluation logs
    timesteps, mean_rewards, mean_ep_lengths = read_eval_logs(file_path)
    
    # Plot the data and save the figure
    plot_eval_data(timesteps, mean_rewards, mean_ep_lengths, output_file)