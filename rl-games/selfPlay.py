import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time, os

class SelfPlayTagEnv(gym.Env):
    """
    Self-play environment that switches between training tagger and runner
    """
    def __init__(self, grid_size=8, max_steps=64, train_tagger=True):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.train_tagger = train_tagger  # True if training tagger, False if training runner

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        obs_size = 4  # Positions: [tagger_x, tagger_y, runner_x, runner_y]
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1,
            shape=(obs_size,), dtype=np.float32
        )

        # Opponent model (will be set externally)
        self.opponent_model = None
        self.opponent_last_action = None

        self.reset()

    def set_opponent(self, model):
        self.opponent_model = model

    def get_opponent_last_action(self):
        return self.opponent_last_action

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly place tagger and runner on different positions
        while True:
            self.tagger_pos = np.array([np.random.randint(0, self.grid_size),
                                        np.random.randint(0, self.grid_size)])
            self.runner_pos = np.array([np.random.randint(0, self.grid_size),
                                        np.random.randint(0, self.grid_size)])

            # Break if positions are different
            if not np.array_equal(self.tagger_pos, self.runner_pos):
                break

        self.steps = 0
        self.tag_made = False

        return self._get_observation(), {}

    def _get_observation(self):
        """Get observation for the training agent"""
        obs = np.concatenate([
            self.tagger_pos.astype(np.float32),
            self.runner_pos.astype(np.float32)
        ])

        return obs

    def _move_agent(self, pos, action):
        """Move agent based on action: 0=up, 1=down, 2=left, 3=right"""
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # Convert action to int if it's a numpy array
        action = int(action)
        new_pos = pos + np.array(moves[action])

        # Clamp to grid boundaries
        new_pos = np.clip(new_pos, 0, self.grid_size-1)
        return new_pos

    def step(self, action):
        """Step with action from training agent, opponent acts via model"""

        if self.train_tagger:
            # Training tagger, opponent is runner
            tagger_action = action

            # Get runner action from opponent model
            if self.opponent_model is not None:
                runner_action, _ = self.opponent_model.predict(self._get_observation(), deterministic=False)
                runner_action = int(runner_action)
            else:
                # Random runner if no opponent model
                runner_action = self.action_space.sample()

            self.opponent_last_action = runner_action
        else:
            # Training runner, opponent is tagger
            runner_action = action

            # Get tagger action from opponent model
            if self.opponent_model is not None:
                tagger_action, _ = self.opponent_model.predict(self._get_observation(), deterministic=False)
                tagger_action = int(tagger_action)
            else:
                # Random tagger if no opponent model
                tagger_action = self.action_space.sample()

            self.opponent_last_action = tagger_action

        # Move both agents
        self.tagger_pos = self._move_agent(self.tagger_pos, tagger_action)
        self.runner_pos = self._move_agent(self.runner_pos, runner_action)

        self.steps += 1

        # Check if tag_made
        self.tag_made = np.array_equal(self.tagger_pos, self.runner_pos)

        # Calculate reward for training agent
        reward = self._calculate_reward()

        # Episode termination
        terminated = self.tag_made or self.steps >= self.max_steps

        return self._get_observation(), reward, terminated, False, {}

    def _calculate_reward(self):
        """Calculate reward for the training agent"""
        if self.train_tagger:
            # tagger rewards
            if self.tag_made:
                return self.max_steps  # Reward for catching the runner

            return -1.0  # Time penalty for tagger
        else:
            # Runner rewards
            if self.tag_made:
                return -self.max_steps  # Penalty for being caught

            return 1.0  # Survival reward for runner

def print_grid(env):
    """Print the current state of the grid"""
    grid = np.full((env.grid_size, env.grid_size), '.', dtype=str)

    # Place agents on grid
    tagger_x, tagger_y = env.tagger_pos
    runner_x, runner_y = env.runner_pos

    # Check if they're on the same position
    if np.array_equal(env.tagger_pos, env.runner_pos):
        grid[tagger_x, tagger_y] = 'X'  # Caught!
    else:
        grid[tagger_x, tagger_y] = 'T'  # tagger
        grid[runner_x, runner_y] = 'R'  # Runner

    # Print grid with borders
    print("  " + " ".join([str(i) for i in range(env.grid_size)]))
    for i in range(env.grid_size):
        print(f"{i} " + " ".join(grid[i]))
    print()

def train_self_play(grid_size=8, max_steps=64, training_rounds=12, timesteps_per_round=1000):
    """Train both agents using self-play"""
    print("Starting self-play training...")

    # Initialize models
    tagger_model = None
    runner_model = None

    tagger_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tag_{grid_size}x{grid_size}_tagger")
    runner_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tag_{grid_size}x{grid_size}_runner")

    # Try to load tagger model
    if os.path.exists(f"{tagger_model_path}.zip"):
        tagger_model = PPO.load(tagger_model_path)
        print(f"Loaded tagger model {tagger_model_path}, previously trained for {tagger_model.num_timesteps} steps")

    # Try to load runner model, create new if doesn't exist
    if os.path.exists(f"{runner_model_path}.zip"):
        runner_model = PPO.load(runner_model_path)
        print(f"Loaded runner model {runner_model_path}, previously trained for {runner_model.num_timesteps} steps")

    for round_num in range(training_rounds):
        start = time.time()

        print(f"\n=== Training Round {round_num + 1}/{training_rounds} ===")

        # Train tagger
        print("Training Tagger...")
        tagger_env = make_vec_env(
            lambda: SelfPlayTagEnv(grid_size=grid_size, max_steps=max_steps, train_tagger=True),
            n_envs=8
        )

        # Set opponent for tagger training
        if runner_model is not None:
            for env in tagger_env.envs:
                env.unwrapped.set_opponent(runner_model)

        if tagger_model is None:
            tagger_model = PPO("MlpPolicy", tagger_env, learning_rate=3e-4, verbose=0)
        else:
            tagger_model.set_env(tagger_env)

        tagger_model.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False)
        tagger_env.close()

        # Train Runner
        print("Training Runner...")
        runner_env = make_vec_env(
            lambda: SelfPlayTagEnv(grid_size=grid_size, max_steps=max_steps, train_tagger=False),
            n_envs=8
        )

        # Set opponent for runner training
        for env in runner_env.envs:
            env.unwrapped.set_opponent(tagger_model)

        if runner_model is None:
            runner_model = PPO("MlpPolicy", runner_env, learning_rate=3e-4, verbose=0)
        else:
            runner_model.set_env(runner_env)

        runner_model.learn(total_timesteps=timesteps_per_round, reset_num_timesteps=False)
        runner_env.close()

        elapsed = time.time() - start
        print(f"Round {round_num + 1} completed in {elapsed:.2f} seconds")

        # Evaluate current performance
        if round_num % 2 == 0:
            print(f"Evaluating after round {round_num + 1}...")
            evaluate_self_play(tagger_model, runner_model, num_episodes=5)

    # Save final models
    tagger_model.save(tagger_model_path)
    runner_model.save(runner_model_path)

    print(f"\nModels saved as {tagger_model_path}.zip and {runner_model_path}.zip")

    print("\nSelf-play training complete!")
    return tagger_model, runner_model

def test_self_play(grid_size = 8, num_episodes=5, render=True, render_fps=1.0):
    tagger_model = PPO.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tag_{grid_size}x{grid_size}_tagger"))
    runner_model = PPO.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tag_{grid_size}x{grid_size}_runner"))
    print(f"Loaded tagger model. Model was trained for {tagger_model.num_timesteps} steps")
    print(f"Loaded runner model. Model was trained for {runner_model.num_timesteps} steps")

    evaluate_self_play(tagger_model, runner_model, num_episodes=num_episodes, render=render, render_fps=render_fps)

def evaluate_self_play(tagger_model=None, runner_model=None, num_episodes=10, render=False, render_fps=1.0):
    """
    Evaluate the performance of trained tagger and runner models.
    This evaluation is ran from the perspective of the tagger agent.
    If no model(s) are provided, will use random actions for one or both agents.
    """

    # Create environment for evaluation
    env = SelfPlayTagEnv()

    # Set opponent models to Runner. Runner will be random if model not provided.
    env.set_opponent(runner_model)

    if render:
        action_names = ['Up', 'Down', 'Left', 'Right']
        print("Legend: T = Tagger, R = Runner, X = Caught, . = Empty Space")
        print("Grid coordinates: rows vertical, columns horizontal\n")

    tagger_wins = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_steps = 0
        rewards = 0

        if render:
            # Print initial state
            print(f"Step {episode_steps}: Initial positions")
            print_grid(env)
            time.sleep(render_fps)

        while True:
            # Determine action for the training agent
            if tagger_model is not None:
                model_action, _ = tagger_model.predict(obs, deterministic=True)
                action = int(model_action)
            else:
                # If no model, use random actions
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            rewards+= reward

            if render:
                print(f"Step {episode_steps}:", end=' ')

                opponent_action = env.get_opponent_last_action()  # Get Runner's last action

                tagger_action = action
                runner_action = opponent_action
                print(f"Rewards: {rewards:.2f} for Tagger.", end=' ')
                print(f"Tagger moves {action_names[tagger_action]}, "
                      f"Runner moves {action_names[runner_action]}.")
                print_grid(env)

            if terminated or truncated:
                if env.tag_made:
                    print(f"🎯 CAUGHT! Tagger wins in {episode_steps} steps!")
                    tagger_wins += 1
                else:
                    print(f"⏰ TIME UP! Runner escapes after {episode_steps} steps!")
                print()

                total_steps += episode_steps
                break

            if render:
                time.sleep(render_fps)

    win_rate = tagger_wins / num_episodes
    avg_steps = total_steps / num_episodes

    print(f"Tagger wins: {tagger_wins}/{num_episodes} ({win_rate:.1%})")
    print(f"Average episode length: {avg_steps:.1f} steps")

    return win_rate

if __name__ == "__main__":
    # evaluate_self_play(render=True)
    train_self_play(grid_size=8, max_steps=64, training_rounds=1, timesteps_per_round=1000)
    # test_self_play(grid_size=8, num_episodes=5, render=True, render_fps=1.0)
