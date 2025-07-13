import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

num_episodes = 10000
max_steps = 100
learning_rate = 0.1
discount_rate = 0.95
epsilon = 1.0

def run_episode(episodes):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_all_episodes = np.zeros(episodes)
    epsilon = 1.0
    epsilon_decay = 0.001
    rng = np.random.default_rng()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
            
            # Take action
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-learning update
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action]
            )
            
            state = new_state
            
            if terminated or truncated:
                break
        
        if reward == 1:
            rewards_all_episodes[episode] = 1
        
        # Epsilon decay
        epsilon = max(0.01, epsilon - epsilon_decay)
        
        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_all_episodes[max(0, episode-999):episode+1])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    
    # Save the trained Q-table
    np.save('q_table.npy', q_table)
    print("Q-table saved as 'q_table.npy'")
    
    # Plot results
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_all_episodes[max(0, t-100):(t+1)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(sum_rewards)
    plt.title('Frozen Lake Q-Learning Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Sum of Rewards (100-episode window)')
    plt.grid(True)
    plt.savefig('frozen_lake4x4.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training completed! Final average reward: {np.mean(rewards_all_episodes[-1000:]):.3f}")
    
    return q_table

def demo_episode():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    print(f"Environment: {env.observation_space.n} states, {env.action_space.n} actions")

    try:
        q_table = np.load('q_table.npy')
        print("Loaded trained Q-table:")
        print(q_table)
        
        # Run a demo episode
        state, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        print(f"\nDemo episode starting from state {state}")
        while (not terminated and not truncated) and steps < max_steps:
            action = np.argmax(q_table[state, :])
            new_state, reward, terminated, truncated, info = env.step(action)
            print(f"Step {steps}: State {state} -> Action {action} -> State {new_state}, Reward: {reward}")
            state = new_state
            steps += 1
            
            if terminated or truncated:
                break
        
        if reward == 1:
            print("🎉 Successfully reached the goal!")
        else:
            print("❌ Failed to reach the goal")
            
    except FileNotFoundError:
        print("No trained Q-table found. Please run training first.")
        return None
        
    env.close()

if __name__ == "__main__":
    print("=== Training Phase ===")
    q_table = run_episode(episodes=10000)
    
    print("\n=== Demo Phase ===")
    demo_episode()
    