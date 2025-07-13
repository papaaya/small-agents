# Reinforcement Learning Games

This folder contains reinforcement learning implementations, starting with the Deep Q-Network (DQN) algorithm for the Frozen Lake environment.

## 🧠 Frozen Lake DQN Algorithm

### Overview
The Frozen Lake DQN implementation demonstrates a complete reinforcement learning solution using neural networks to learn optimal navigation policies in a 4x4 grid environment.

### Deep Q-Network (DQN) Algorithm Steps

#### **Step 1: Environment Setup**
- Create a 4x4 Frozen Lake grid where the agent (S) must navigate from start to goal (G)
- States: 16 possible positions (0-15), Actions: 4 directions (Left, Down, Right, Up)
- Rewards: +1 for reaching goal, 0 for falling in holes or other actions

#### **Step 2: Neural Network Architecture**
- **Policy Network**: Takes current state as input, outputs Q-values for all 4 actions
- **Target Network**: Copy of policy network (updated less frequently for stability)
- Architecture: Input layer (16 nodes) → Hidden layer (16 nodes, ReLU) → Output layer (4 nodes)

#### **Step 3: Experience Replay Buffer**
- Store experiences as (state, action, reward, next_state, done) tuples
- Randomly sample batches of experiences for training (prevents correlation issues)
- Buffer size: 1000 experiences, Batch size: 32

#### **Step 4: Action Selection (Epsilon-Greedy)**
- **Exploration**: With probability ε (epsilon), choose random action
- **Exploitation**: With probability (1-ε), choose action with highest Q-value
- Start with ε = 1.0 (100% random), gradually decay to ε = 0.01

#### **Step 5: Experience Collection**
- Execute selected action in environment
- Observe new state, reward, and whether episode ended
- Store experience in replay buffer
- Continue until episode terminates (goal reached or hole fallen)

#### **Step 6: Q-Learning Update**
- Sample random batch from replay buffer
- For each experience, calculate target Q-value:
  - **If terminated**: Target = reward
  - **If not terminated**: Target = reward + γ × max(Q(next_state))
- γ (gamma) = 0.9 (discount factor for future rewards)

#### **Step 7: Neural Network Training**
- Compute loss between current Q-values and target Q-values
- Use Mean Squared Error (MSE) loss function
- Update policy network weights using Adam optimizer
- Learning rate: 0.001

#### **Step 8: Target Network Synchronization**
- Every 10 steps, copy policy network weights to target network
- This prevents the "moving target" problem and stabilizes training

#### **Step 9: Epsilon Decay**
- Gradually reduce exploration rate: ε = max(ε - 1/episodes, 0.01)
- Balance exploration vs exploitation as agent learns

#### **Step 10: Convergence**
- Train for 10,000 episodes
- Agent learns optimal policy to navigate from start to goal
- Save trained model and visualize learning progress

### Key Innovations
The key innovation of DQN is using a neural network to approximate Q-values instead of a lookup table, allowing it to handle large state spaces efficiently while using experience replay and target networks to ensure stable learning.

### Files

#### Implementation Files
- `frozen_lake_dqn.py`: Main DQN implementation with training and testing
- `frozen_lake_qtable.py`: Traditional Q-table implementation for comparison

#### Generated Files
- `frozen_lake_dql.pt`: Trained neural network model
- `frozen_lake_dql.png`: Learning progress visualization (rewards and epsilon decay)
- `q_table.npy`: Saved Q-table for traditional Q-learning

### Usage

#### Training the DQN Agent
```bash
python frozen_lake_dqn.py
```

This will:
1. Train the DQN agent for 10,000 episodes
2. Save the trained model as `frozen_lake_dql.pt`
3. Generate learning progress plots as `frozen_lake_dql.png`
4. Test the trained agent for 100 episodes

#### Hyperparameters
- **Episodes**: 10,000
- **Max Steps**: 100 per episode
- **Learning Rate**: 0.001
- **Discount Rate**: 0.9
- **Epsilon**: 1.0 → 0.01 (decay)
- **Memory Size**: 1000 experiences
- **Batch Size**: 32
- **Target Update**: Every 10 steps

### Environment Details

#### Frozen Lake 4x4
```
SFFF
FHFH
FFFH
HFFG
```
- **S**: Start (safe)
- **F**: Frozen surface (safe)
- **H**: Hole (fall through)
- **G**: Goal (reward = 1)

#### Actions
- **0**: Left
- **1**: Down
- **2**: Right
- **3**: Up

#### States
- 16 possible positions (0-15)
- One-hot encoded for neural network input

### Algorithm Comparison

#### DQN vs Q-Table
| Feature | DQN | Q-Table |
|---------|-----|---------|
| State Representation | Neural Network | Lookup Table |
| Memory Usage | Efficient | Exponential |
| Scalability | High | Low |
| Training Stability | Experience Replay + Target Network | Direct Updates |
| Convergence | Slower but more robust | Faster for small spaces |

### Learning Visualization

The training process generates plots showing:
1. **Average Rewards**: Rolling average of successful episodes over time
2. **Epsilon Decay**: Exploration rate reduction over episodes

### Future Enhancements

Potential improvements to explore:
- **Double DQN**: Separate networks for action selection and evaluation
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important experiences more frequently
- **Multi-step Learning**: Use n-step returns for better credit assignment
- **Rainbow DQN**: Combine multiple DQN improvements

### Dependencies
- `gymnasium`: Reinforcement learning environments
- `torch`: Deep learning framework
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `collections.deque`: Efficient experience replay buffer 