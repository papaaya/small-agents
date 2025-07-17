# Technical Documentation

This folder contains detailed technical documentation for the Small Agents project, including architecture diagrams, sequence flows, and implementation details.

## 📋 Documentation Index

### 🏗️ Architecture
- [Agent Architecture Overview](architecture/agent-architecture.md) - High-level system design
- [Tool Design Patterns](architecture/tool-patterns.md) - Reusable tool patterns
- [Grid Game Framework](architecture/grid-game-framework.md) - Generic pattern for grid-based problems

### 🔄 Sequence Diagrams
- [General Chat Agent Flow](sequence-diagrams/general-chat-agent.md) - Interactive conversation flow
- [ARC Agent Reasoning](sequence-diagrams/arc-agent-reasoning.md) - Visual reasoning process
- [Frozen Lake RL Training](sequence-diagrams/frozen-lake-training.md) - Reinforcement learning flow
- [Tool Execution Patterns](sequence-diagrams/tool-execution.md) - Standard tool execution flow
- [Grid Game Patterns](sequence-diagrams/grid-game-patterns.md) - Navigation, puzzle solving, and visual reasoning flows

### 🛠️ Implementation Guides
- [Building Custom Agents](implementation/custom-agents.md) - Step-by-step guide
- [Adding New Tools](implementation/adding-tools.md) - Tool development patterns
- [Observability Setup](implementation/observability.md) - Logfire integration guide

### 📊 Performance & Monitoring
- [Performance Metrics](performance/metrics.md) - Key performance indicators
- [Debugging Guide](performance/debugging.md) - Troubleshooting common issues
- [Optimization Tips](performance/optimization.md) - Performance best practices

## 🎯 Quick Reference

### Common Patterns
```python
# Tool definition pattern
@agent.tool
async def my_tool(ctx: RunContext, param: str) -> str:
    with logfire.span("my_tool") as span:
        # Implementation
        span.set_attribute("result", result)
        return result

# Agent configuration pattern
agent = Agent(
    model="gemini-2.0-flash-exp",
    system_prompt="...",
    output_type=MyOutputType
)
```

### Grid Game Framework
```python
# Pattern 1: Navigation & Pathfinding
player_pos = await find_player_position(grid)
goal_pos = await find_goal_position(grid)
is_valid = await validate_move(from_pos, action)
updated_grid = await move_player_agent(grid, action)

# Pattern 2: Puzzle Solving & State Transformation
state_analysis = await analyze_grid_state(grid)
is_valid = await valid_action(action)
updated_grid = await update_grid(grid, action)
reward = await get_reward(state, action)

# Pattern 3: Visual Reasoning & Pattern Recognition
analysis = await analyze_grid(grid, "input")
transformed = await apply_transformation(grid, "rotate_90")
is_valid = await compare_grids(transformed, expected_output)
```

## 🔗 Related Documentation
- [Main README](../README.md) - Project overview and quick start
- [Requirements](../requirements.txt) - Dependencies
- [Source Code](../src/) - Implementation files 