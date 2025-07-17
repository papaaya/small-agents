# Frozen Lake - Reinforcement Learning Training Flow

## Overview
The Frozen Lake DQN implementation demonstrates a complete reinforcement learning solution using neural networks to learn optimal navigation policies. This document shows the detailed training flow.

## Main Training Loop

```mermaid
sequenceDiagram
    participant E as Environment
    participant A as Agent
    participant N as Neural Network
    participant B as Replay Buffer
    participant T as Target Network
    participant L as Logfire

    Note over E,L: Episode Initialization
    
    E->>A: Reset environment (state = 0)
    A->>L: Create episode span
    A->>A: Initialize episode_reward = 0
    
    loop Episode Steps
        A->>N: Get Q-values for current state
        N->>A: Return Q-values [Q(s,a1), Q(s,a2), Q(s,a3), Q(s,a4)]
        
        A->>A: Epsilon-greedy action selection
        alt Exploration (ε probability)
            A->>A: Choose random action
        else Exploitation (1-ε probability)
            A->>A: Choose action with max Q-value
        end
        
        A->>E: Execute action
        E->>A: Return (next_state, reward, done)
        
        A->>B: Store experience (state, action, reward, next_state, done)
        A->>A: Update episode_reward += reward
        
        A->>A: Update current_state = next_state
        
        alt Episode not done
            A->>A: Continue to next step
        else Episode done
            A->>L: Log episode metrics (reward, steps)
            A->>A: Break episode loop
        end
    end
    
    Note over E,L: Training Phase
    
    alt Training condition met
        A->>B: Sample batch of experiences
        B->>A: Return batch [(s,a,r,s',d), ...]
        
        loop For each experience in batch
            A->>N: Get Q-values for current state
            N->>A: Return current Q-values
            
            A->>T: Get Q-values for next state
            T->>A: Return target Q-values
            
            A->>A: Calculate target Q-value
            alt Episode done
                A->>A: target = reward
            else Episode not done
                A->>A: target = reward + γ * max(Q(s'))
            end
            
            A->>A: Calculate loss = MSE(current_Q, target)
            A->>N: Update network weights
            N->>A: Return updated weights
        end
        
        A->>L: Log training metrics (loss, batch_size)
    end
    
    A->>A: Update epsilon (decay exploration)
    A->>L: Log epsilon value
```

## Neural Network Architecture

```mermaid
sequenceDiagram
    participant I as Input Layer
    participant H as Hidden Layer
    participant O as Output Layer
    participant L as Logfire

    I->>L: Create network span
    
    I->>I: Input state (16-dimensional one-hot)
    I->>L: Set input attributes (state_id)
    
    I->>H: Linear transformation (16 → 16)
    H->>H: Apply ReLU activation
    H->>L: Set hidden_activation attributes
    
    H->>O: Linear transformation (16 → 4)
    O->>O: Output Q-values for all actions
    O->>L: Set output attributes (q_values)
    
    O->>L: Log forward pass completion
```

## Experience Replay Buffer

```mermaid
sequenceDiagram
    participant A as Agent
    participant B as Replay Buffer
    participant L as Logfire

    A->>B: Store experience (s, a, r, s', d)
    B->>L: Create storage span
    
    B->>B: Add experience to buffer
    B->>L: Set buffer_size attribute
    
    alt Buffer full
        B->>B: Remove oldest experience
        B->>L: Log buffer overflow
    end
    
    B->>A: Confirm storage
    
    Note over A,B: Sampling Phase
    
    A->>B: Request batch of size 32
    B->>L: Create sampling span
    
    B->>B: Randomly sample experiences
    B->>L: Set batch_size attribute
    
    B->>A: Return batch of experiences
    A->>L: Log sampling completion
```

## Target Network Synchronization

```mermaid
sequenceDiagram
    participant P as Policy Network
    participant T as Target Network
    participant L as Logfire

    Note over P,T: Initial Setup
    
    P->>T: Copy initial weights
    T->>L: Log initial sync
    
    loop Every 10 steps
        P->>P: Update weights during training
        P->>L: Log policy updates
        
        alt Sync condition met
            P->>T: Copy current weights
            T->>L: Log target sync
            T->>L: Set sync_interval attribute
        end
    end
```

## Epsilon Decay Strategy

```mermaid
sequenceDiagram
    participant A as Agent
    participant L as Logfire

    A->>A: Initialize epsilon = 1.0
    A->>L: Log initial epsilon
    
    loop After each episode
        A->>A: Calculate new epsilon
        A->>A: epsilon = max(epsilon - decay_rate, min_epsilon)
        
        A->>L: Log epsilon update
        A->>L: Set exploration_rate attribute
        
        alt Epsilon near minimum
            A->>L: Log exploration phase ending
        end
    end
```

## Performance Monitoring

```mermaid
sequenceDiagram
    participant A as Agent
    participant L as Logfire
    participant M as Metrics

    A->>L: Create training session span
    
    loop Every episode
        A->>M: Calculate episode metrics
        M->>A: Return (reward, steps, success)
        A->>L: Log episode metrics
        
        A->>M: Calculate running averages
        M->>A: Return (avg_reward, avg_steps, success_rate)
        A->>L: Log running averages
    end
    
    loop Every training batch
        A->>M: Calculate training metrics
        M->>A: Return (loss, gradient_norm)
        A->>L: Log training metrics
    end
    
    A->>L: Log final performance summary
```

## Key Components

### 1. Environment
- **State Space**: 16 discrete states (0-15)
- **Action Space**: 4 actions (Left, Down, Right, Up)
- **Rewards**: +1 for goal, 0 for holes/other
- **Termination**: Goal reached or hole fallen

### 2. Neural Network
- **Input Layer**: 16 nodes (one-hot state encoding)
- **Hidden Layer**: 16 nodes with ReLU activation
- **Output Layer**: 4 nodes (Q-values for each action)
- **Optimizer**: Adam with learning rate 0.001

### 3. Experience Replay
- **Buffer Size**: 1000 experiences
- **Batch Size**: 32 experiences
- **Sampling**: Uniform random sampling
- **Replacement**: Oldest experiences removed when full

### 4. Target Network
- **Update Frequency**: Every 10 steps
- **Purpose**: Stabilize training by providing fixed targets
- **Method**: Direct weight copying from policy network

### 5. Exploration Strategy
- **Initial Epsilon**: 1.0 (100% random actions)
- **Final Epsilon**: 0.01 (1% random actions)
- **Decay Rate**: Linear decay over episodes
- **Purpose**: Balance exploration vs exploitation

## Training Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.001 | Controls weight update magnitude |
| Discount Factor (γ) | 0.9 | Future reward importance |
| Buffer Size | 1000 | Experience storage capacity |
| Batch Size | 32 | Training batch size |
| Target Update | 10 | Target network sync frequency |
| Episodes | 10,000 | Total training episodes |
| Initial Epsilon | 1.0 | Starting exploration rate |
| Final Epsilon | 0.01 | Minimum exploration rate |

## Performance Metrics

- **Episode Reward**: Total reward per episode
- **Success Rate**: Percentage of episodes reaching goal
- **Average Steps**: Mean steps to goal/successful episodes
- **Training Loss**: MSE between predicted and target Q-values
- **Exploration Rate**: Current epsilon value
- **Convergence**: When success rate stabilizes above 90%

## Common Training Patterns

### 1. Early Training
- High exploration (ε ≈ 1.0)
- Low success rate (< 20%)
- High training loss
- Random exploration dominates

### 2. Mid Training
- Decreasing exploration (ε ≈ 0.5)
- Improving success rate (20-80%)
- Decreasing training loss
- Learning from successful experiences

### 3. Late Training
- Low exploration (ε ≈ 0.01)
- High success rate (> 90%)
- Stable training loss
- Exploitation of learned policy 