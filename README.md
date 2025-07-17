# Small Agents: Building Intelligent AI Agents with Pydantic-AI

> *A journey into creating intelligent agents that can solve complex problems, play games, and reason about the world around them.*

## 📋 Table of Contents

### 🚀 Getting Started
- [Quick Start](#-quick-start)
- [What We're Building](#-what-were-building)

### 🤖 Agent Showcase
- [Weather Agent](#1-weather-agent---real-world-data-integration)
- [General Chat Agent](#2-general-chat-agent---conversational-intelligence-)
- [Frozen Lake Agent](#3-frozen-lake-agent---reinforcement-learning-in-action)
- [ARC Agent](#4-arc-agent---abstract-reasoning-)

### 🎮 Grid Game Patterns
- [Navigation & Pathfinding](#pattern-1-navigation--pathfinding-frozen-lake-style)
- [Puzzle Solving & State Transformation](#pattern-2-puzzle-solving--state-transformation-sokoban-style)
- [Visual Reasoning & Pattern Recognition](#pattern-3-visual-reasoning--pattern-recognition-arc-style)
- [Key Learnings from Implementation](#key-learnings-from-implementation)

### 🔍 Deep Dives
- [ARC Agent Architecture](#-deep-dive-arc-agent-architecture)
- [Observability with Logfire](#-observability-with-logfire)

### 🛠️ Development
- [Building Your Own Agent](#-building-your-own-agent)
- [Key Learnings](#-key-learnings)
- [What's Next](#-whats-next)
- [Contributing](#-contributing)

### 📚 Documentation
- [Technical Documentation](technical_docs/README.md)
- [Architecture Guides](technical_docs/architecture/)
- [Sequence Diagrams](technical_docs/sequence-diagrams/)
- [Implementation Guides](technical_docs/implementation/)

### ⚡ Quick Reference for Developers
```bash
# Run agents
python src/01_weather_agent.py          # Weather queries
python src/04_general_chat_agent.py     # Interactive chat
python src/10_frozen_lake_agent.py      # Navigation game
python src/14_simple_arc_agent.py       # Visual reasoning

# Grid game patterns
python src/012_frozen_lake_coordinate_approach.py  # Navigation pattern
python src/07_LMGAME_agent.py                      # Puzzle solving pattern

# Reinforcement learning
python rl-games/frozen_lake_dqn.py      # DQN implementation
```

## 🎯 What We're Building

This repository showcases the evolution of AI agents - from simple weather queries to complex reasoning systems that can solve ARC (Abstraction and Reasoning Corpus) challenges. Each agent demonstrates different aspects of intelligent behavior and provides reusable patterns for building your own AI systems.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd small-agents
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment
cp .env.example .env  # Add your API keys
```

## 📁 Project Structure

```
small-agents/
├── src/                          # Main agent implementations
│   ├── 01_weather_agent.py      # Weather data integration
│   ├── 04_general_chat_agent.py # Conversational AI
│   ├── 10_frozen_lake_agent.py  # Navigation game
│   ├── 14_simple_arc_agent.py   # Visual reasoning
│   └── 07_LMGAME_agent.py       # Puzzle solving
├── rl-games/                     # Reinforcement learning
│   ├── frozen_lake_dqn.py       # Deep Q-Network
│   └── frozen_lake_qtable.py    # Q-Learning
├── technical_docs/               # Architecture & design docs
│   ├── architecture/            # System design patterns
│   ├── sequence-diagrams/       # Flow diagrams
│   └── implementation/          # Development guides
├── prompts/                      # Prompt management
└── requirements.txt             # Dependencies
```

## 🤖 Agent Showcase

### Agent Types Overview

| Agent Type | Best For | Complexity | Example |
|------------|----------|------------|---------|
| **Data Integration** | External APIs, real-time data | Low | Weather Agent |
| **Conversational** | Chat, memory, multi-turn | Medium | General Chat Agent |
| **Navigation** | Pathfinding, obstacle avoidance | Medium | Frozen Lake Agent |
| **Puzzle Solving** | Logic games, state transformation | High | Sokoban Agent |
| **Visual Reasoning** | Pattern recognition, transformations | High | ARC Agent |
| **Reinforcement Learning** | Learning from experience | High | DQN Agent |

### 1. **Weather Agent** - Real-world Data Integration
*"How's the weather in Tokyo?"*

Our first agent demonstrates how to integrate external APIs and provide real-time information. It shows the pattern of:
- Tool-based data fetching
- Error handling and retries
- Structured output validation

```bash
python src/01_weather_agent.py
```

### 2. **General Chat Agent** - Conversational Intelligence ⭐
*"Let's have a meaningful conversation"*

This is our most sophisticated conversational agent, featuring:
- **Memory**: Remembers your name and preferences
- **Tools**: Can calculate, tell time, and think step-by-step
- **Interactive Mode**: Real-time conversation with command support
- **Observability**: Comprehensive logging with Logfire

```bash
# Demo mode - see all capabilities
python src/04_general_chat_agent.py

# Interactive mode - chat live
python src/04_general_chat_agent.py --interactive
```

### 3. **Frozen Lake Agent** - Reinforcement Learning in Action
*"Can AI learn to navigate a dangerous environment?"*

A complete RL implementation using Deep Q-Networks:
- Neural network-based learning
- Experience replay for stability
- Visual learning progress tracking

```bash
python rl-games/frozen_lake_dqn.py
```

### 4. **ARC Agent** - Abstract Reasoning 🧠
*"Can AI solve visual reasoning puzzles?"*

Our most advanced agent tackles the Abstraction and Reasoning Corpus:
- **Grid Analysis**: Understanding spatial patterns
- **Transformation DSL**: Domain-specific language for operations
- **Step-by-step Reasoning**: Breaking complex problems into solvable pieces
- **Confidence Scoring**: Self-assessment of solutions

```bash
python src/14_simple_arc_agent.py
```

## 🎮 Grid Game Patterns: From Navigation to Puzzle Solving

After building multiple grid-based agents, we've identified **three powerful patterns** that can be applied across different domains:

### Pattern Comparison

| Aspect | Navigation | Puzzle Solving | Visual Reasoning |
|--------|------------|----------------|------------------|
| **Primary Goal** | Reach destination | Transform state | Find patterns |
| **Key Tools** | Position tracking | State validation | Pattern analysis |
| **Complexity** | Medium | High | High |
| **Examples** | Frozen Lake | Sokoban | ARC challenges |
| **Success Metric** | Goal reached | Puzzle solved | Pattern matched |

### Pattern 1: Navigation & Pathfinding (Frozen Lake Style)

**Best for**: Navigation games, pathfinding, exploration problems

```python
# Core Components
@agent.tool
async def find_player_position(ctx: RunContext, grid: List[List[str]]) -> tuple:
    """Locate the player/agent in the grid"""
    
@agent.tool  
async def find_goal_position(ctx: RunContext, grid: List[List[str]]) -> tuple:
    """Find the target destination"""
    
@agent.tool
async def validate_move(ctx: RunContext, from_pos: tuple, action: str) -> MoveValidation:
    """Check if a move is valid and safe"""
    
@agent.tool
async def move_player_agent(ctx: RunContext, grid: List[List[str]], action: str) -> List[List[str]]:
    """Execute the move and update grid state"""
```

**The Navigation Loop**:
1. **Locate** → Find current position and goal
2. **Plan** → Determine safe path avoiding obstacles
3. **Validate** → Check if move is legal
4. **Execute** → Move and update state
5. **Repeat** → Continue until goal reached

### Pattern 2: Puzzle Solving & State Transformation (Sokoban Style)

**Best for**: Logic puzzles, state-based games, transformation problems

```python
# Core Components
@agent.tool
async def analyze_grid_state(ctx: RunContext, grid: List[List[str]]) -> GridAnalysis:
    """Understand current state and identify key elements"""
    
@agent.tool
async def valid_action(ctx: RunContext, action: Action) -> str:
    """Check if action follows game rules"""
    
@agent.tool
async def update_grid(ctx: RunContext, grid: List[List[str]], action: Action) -> List[List[str]]:
    """Apply action and transform grid state"""
    
@agent.tool
async def get_reward(ctx: RunContext, state: str, action: Action) -> float:
    """Calculate reward for action to guide learning"""
    
@agent.tool
async def verify_solution(ctx: RunContext) -> str:
    """Check if puzzle is solved"""
```

**The Puzzle Solving Loop**:
1. **Analyze** → Understand current state and constraints
2. **Hypothesize** → Propose valid actions
3. **Transform** → Apply action and update state
4. **Evaluate** → Calculate reward and check progress
5. **Iterate** → Continue until puzzle solved

### Pattern 3: Visual Reasoning & Pattern Recognition (ARC Style)

**Best for**: Visual puzzles, pattern matching, abstract reasoning

```python
# Core Components
@agent.tool
async def analyze_grid(ctx: RunContext, grid: List[List[int]], name: str = "grid") -> GridAnalysis:
    """Understand the structure and patterns in a grid"""
    
@agent.tool
async def apply_transformation(ctx: RunContext, grid: List[List[int]], operation: str, **params) -> List[List[int]]:
    """Apply geometric or value transformations"""
    
@agent.tool
async def compare_grids(ctx: RunContext, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Compare grids and validate solutions"""
```

**The Reasoning Loop**:
1. **Analyze** → Understand the current state
2. **Hypothesize** → Propose transformations
3. **Apply** → Execute the transformation
4. **Validate** → Check if result matches expectation
5. **Iterate** → Repeat until solution found

### Why These Patterns Work

- **Modular**: Each tool has a single responsibility
- **Composable**: Tools can be combined in any order
- **Observable**: Every step is logged and traceable
- **Extensible**: Easy to add new operations or analysis types
- **Domain-Specific**: Each pattern is optimized for its problem type

### Key Learnings from Implementation

#### From Frozen Lake Agents
- **Position Tracking**: Always maintain clear coordinate systems (1-based vs 0-based)
- **Boundary Validation**: Check bounds before every move to prevent crashes
- **State Visualization**: Pretty-print grids for debugging and agent understanding
- **Reward Shaping**: Use meaningful rewards (-1000 for holes, +100 for goal) to guide learning
- **Conversation Memory**: Maintain conversation history for multi-step reasoning

#### From LMGAME (Sokoban) Agent
- **State Transformation**: Deep copy grids to avoid mutation issues
- **Complex Rule Validation**: Handle edge cases like box pushing and wall collisions
- **Reward Calculation**: Dynamic reward systems that consider progress and deadlocks
- **Restart Mechanisms**: Ability to reset to original state when stuck
- **Prompt Versioning**: Use different prompt versions for different puzzle complexities

## 🔍 Deep Dive: ARC Agent Architecture

The ARC agent demonstrates our most sophisticated reasoning system:

### Problem Decomposition
```
Input Grid → Analysis → Pattern Recognition → Transformation Planning → Execution → Validation
```

### Key Innovations

1. **Domain-Specific Language (DSL)**: A set of primitive operations that can be combined
2. **Confidence Scoring**: The agent assesses its own confidence in solutions
3. **Step-by-step Reasoning**: Each transformation is documented and explained
4. **Multi-example Learning**: Uses multiple training examples to generalize patterns

### Example ARC Solution
```python
# Agent identifies a rotation pattern
transformation_steps = [
    TransformationStep(
        operation="rotate_90",
        description="Rotate the grid 90 degrees clockwise"
    ),
    TransformationStep(
        operation="replace_values", 
        parameters={"old_value": 1, "new_value": 2},
        description="Replace all 1s with 2s"
    )
]
```

## 📊 Observability with Logfire

Every agent includes comprehensive logging:

```python
# Automatic instrumentation
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# Custom spans for business logic
with logfire.span("analyze_grid") as span:
    span.set_attribute("grid_shape", grid.shape)
    span.set_attribute("unique_values", values)
```

**What We Track:**
- Tool execution times and success rates
- Agent reasoning patterns and confidence scores
- Error rates and debugging information
- User interaction flows

## 🛠️ Building Your Own Agent

### 🎯 Which Agent Should You Start With?

**Beginner** → Start with Weather Agent or General Chat Agent
- Learn basic tool integration and conversation flow
- Understand Pydantic-AI fundamentals

**Intermediate** → Try Frozen Lake or Navigation patterns
- Master grid-based reasoning
- Learn position tracking and validation

**Advanced** → Explore Puzzle Solving or Visual Reasoning
- Complex state transformations
- Pattern recognition and confidence scoring

**Expert** → Dive into Reinforcement Learning
- Neural networks and experience replay
- Advanced learning algorithms

### Step 1: Define Your Tools
```python
@agent.tool
async def my_tool(ctx: RunContext, input_data: str) -> str:
    """What your tool does"""
    with logfire.span("my_tool") as span:
        # Your logic here
        result = process_data(input_data)
        span.set_attribute("result", result)
        return result
```

### Step 2: Create Your Agent
```python
agent = Agent(
    model="gemini-2.0-flash-exp",
    system_prompt="You are a helpful assistant...",
    output_type=MyOutputType
)
```

### Step 3: Add Observability
```python
# Automatic logging of all operations
logfire.instrument_pydantic_ai()

# Custom metrics
logfire.info("Agent started", agent_type="my_agent")
```

### Step 4: Test and Iterate
```python
result = await agent.run("Your prompt here")
print(f"Confidence: {result.output.confidence}")
```

## 🎯 Key Learnings

### 1. **Tool Design Matters**
- Tools should be atomic and composable
- Clear input/output contracts prevent errors
- Good tool descriptions help the agent choose correctly

### 2. **Observability is Essential**
- Logfire provides insights into agent behavior
- Spans help debug complex reasoning chains
- Metrics reveal performance bottlenecks

### 3. **Pattern Recognition is Powerful**
- The grid analysis pattern works across many domains
- Breaking problems into steps improves success rates
- Validation at each step catches errors early

### 4. **Confidence Scoring Helps**
- Agents can assess their own reliability
- Low confidence triggers different strategies
- Helps users understand when to trust the agent

## 🚀 What's Next?

We're exploring:
- **Multi-agent systems** - Agents that collaborate
- **Learning from feedback** - Improving based on user corrections
- **Visual reasoning** - Processing images and diagrams
- **Planning and execution** - Breaking complex tasks into steps
- **Hybrid patterns** - Combining navigation, puzzle solving, and visual reasoning
- **Real-time adaptation** - Dynamic pattern switching based on problem complexity

## 🤝 Contributing

We'd love your help! Here's how to contribute:

1. **Try the agents** - Run them and share your experiences
2. **Add new tools** - Extend existing agents with new capabilities
3. **Create new agents** - Build agents for new domains
4. **Improve documentation** - Help others understand the patterns

## 📚 Resources

- [Pydantic-AI Documentation](https://docs.pydantic-ai.com/)
- [Logfire Observability](https://logfire.com/)
- [ARC Challenge](https://github.com/fchollet/ARC)
- [Reinforcement Learning Basics](https://spinningup.openai.com/)

---

*Built with ❤️ using Pydantic-AI, Logfire, and a lot of curiosity about what AI can do.*

For detailed technical documentation, see the [technical_docs/](technical_docs/) folder. 