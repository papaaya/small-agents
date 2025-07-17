# Small Agents: Building Intelligent AI Agents with Pydantic-AI

> *A journey into creating intelligent agents that can solve complex problems, play games, and reason about the world around them.*

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

## 🤖 Agent Showcase

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

## 🎮 The Generic Agentic Pattern for Grid Games

After building multiple agents, we've identified a powerful pattern for solving grid-based problems. Here's the framework:

### Core Components

```python
# 1. Grid Analysis Tools
@agent.tool
async def analyze_grid(ctx: RunContext, grid: List[List[int]], name: str = "grid") -> GridAnalysis:
    """Understand the structure and patterns in a grid"""
    # Analyze shape, values, patterns
    # Return structured analysis

# 2. Transformation Tools  
@agent.tool
async def apply_transformation(ctx: RunContext, grid: List[List[int]], operation: str, **params) -> List[List[int]]:
    """Apply geometric or value transformations"""
    # Execute operations like rotate, flip, replace
    # Return transformed grid

# 3. Validation Tools
@agent.tool
async def compare_grids(ctx: RunContext, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Compare grids and validate solutions"""
    # Check if transformation produces expected result
```

### The Reasoning Loop

1. **Analyze** → Understand the current state
2. **Hypothesize** → Propose transformations
3. **Apply** → Execute the transformation
4. **Validate** → Check if result matches expectation
5. **Iterate** → Repeat until solution found

### Why This Pattern Works

- **Modular**: Each tool has a single responsibility
- **Composable**: Tools can be combined in any order
- **Observable**: Every step is logged and traceable
- **Extensible**: Easy to add new operations or analysis types

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