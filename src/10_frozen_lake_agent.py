"""
Lets import frozen lake from gymnasium and create a simple pydantic-ai LLM agent that can play the game.
"""

import asyncio
import gymnasium as gym
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import Literal, List, Tuple, Optional, cast
from typing import Tuple
import logfire
import dotenv

dotenv.load_dotenv()

# Configure logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)

class Action(BaseModel):
    action: Literal["left", "right", "up", "down"]
    
    def to_gym_action(self) -> int:
        """Convert to gymnasium action integer"""
        gym_map = {"left": 0, "down": 1, "right": 2, "up": 3}
        return gym_map[self.action]

class AgentDecision(BaseModel):
    action: Action = Field(description="The action the agent chooses to take")
    projected_reward: float = Field(description="The projected reward for this action")

class Position(BaseModel):
    row: int = Field(description="Row index (0-based)")
    col: int = Field(description="Column index (0-based)")

class GridAnalysisResult(BaseModel):
    agent_position: Position = Field(description="Current position of the agent")
    goal_position: Position = Field(description="Position of the goal")
    holes_positions: List[Position] = Field(description="Positions of all holes")
    safe_positions: List[Position] = Field(description="Positions of safe frozen lake tiles")

class LookAheadHolesResult(BaseModel):
    action_list: list[Action] = Field(description="The list of actions to avoid the hole, hence LLM agent should not take this action")
    direction: str = Field(description="The direction where the hole was found")
    thought_hole_direction: str = Field(description="The direction the agent should look ahead to see if there are any holes in the grid")
    agent_position: Position = Field(description="Current position of the agent")
    holes_found: List[Position] = Field(description="Positions of holes found in the analysis")
    safe_directions: List[str] = Field(description="Directions that are safe to move")

class RewardUpdateResult(BaseModel):
    best_action: Action = Field(description="The best action the agent should take based on analysis")
    projected_reward: float = Field(description="The projected reward for the best action")
    reasoning: str = Field(description="Reasoning for the chosen action and reward")

class GameState(BaseModel):
    state: int = Field(description="The current state of the environment")
    action: Action | None = Field(description="The action to take", default=None)
    reward: float = Field(description="The reward for the action")
    done: bool = Field(description="Whether the game is over")
    truncated: bool = Field(description="Whether the game is truncated")
    rendered_grid: str = Field(description="A text rendering of the current environment grid")
    total_reward: float = Field(description="The total reward for the agent")

def parse_grid(grid_str: str) -> Tuple[List[List[str]], Optional[Position], Optional[Position], List[Position]]:
    """
    Parse the grid string and extract key positions.
    
    Args:
        grid_str: The grid as a string with newlines
        
    Returns:
        Tuple of (grid_matrix, agent_position, goal_position, holes_positions)
    """
    lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
    grid = [list(line) for line in lines]
    
    agent_pos = None
    goal_pos = None
    holes = []
    
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == 'P':
                agent_pos = Position(row=row_idx, col=col_idx)
            elif cell == 'G':
                goal_pos = Position(row=row_idx, col=col_idx)
            elif cell == 'H':
                holes.append(Position(row=row_idx, col=col_idx))
    
    return grid, agent_pos, goal_pos, holes

def get_adjacent_positions(pos: Position, grid: List[List[str]]) -> List[Tuple[str, Position, str]]:
    """
    Get all adjacent positions and their contents.
    
    Args:
        pos: Current position
        grid: The grid matrix
        
    Returns:
        List of (direction, position, content) tuples
    """
    directions = [
        ("up", Position(row=pos.row-1, col=pos.col)),
        ("down", Position(row=pos.row+1, col=pos.col)),
        ("left", Position(row=pos.row, col=pos.col-1)),
        ("right", Position(row=pos.row, col=pos.col+1))
    ]
    
    adjacent = []
    for direction, adj_pos in directions:
        if (0 <= adj_pos.row < len(grid) and 
            0 <= adj_pos.col < len(grid[0])):
            content = grid[adj_pos.row][adj_pos.col]
            adjacent.append((direction, adj_pos, content))
    
    return adjacent

agent = Agent(
    "openai:o3-mini",
    output_type=AgentDecision,
    system_prompt="""
You are an expert in the game of 2D grid world. You will be given the Frozen Lake environment from gymnasium.
You have to reach the goal 'G' in the grid, and your starting position is 'P'. You can move in 4 directions: left, right, up, down.
Use `analyze_grid` tool to get detailed information about the current grid state.
Use `look_ahead_holes` tool to analyze the grid and find holes around the player's position to make a good decision.
Use `update_reward` tool to update the reward for the agent and get the best action to take.

# Symbols in the grid
- 'P': Your current position (the player/agent)
- 'S': Starting position (where you begin)
- 'F': Frozen lake (safe to step on)
- 'H': Hole (dangerous, game ends if you fall in)
    - Get a reward of -1000 if you fall in a hole 'H'
- 'G': Goal (the target position where the player/agent should reach)
    - Get a reward of 100 if you reach the goal 'G'

# Strategy and tips
- At every step please look at Goal 'G' in the grid, the player/agent should reach closer to the goal after every step.
- Use <think> </think> tags to think about your action. 
    - In <think> tags, write information about the look_ahead_holes tool to look ahead to see if there are any holes in the grid.
    - Use <think> tags to analyze the neighboring cells of the agent's/player's current position.
- You must avoid the holes in the grid. represented as 'H' in the grid.
- You must reach the goal. represented as 'G' in the grid.
- You must avoid the walls or boundaries of the grid. 

# Enhanced Spatial aware tools usage
These tools are used to analyze the grid and the player's position to make a good decision.

## analyze_grid tool
- Use this tool first to get detailed information about the current grid state
- This tool will return:
    - agent_position: Your current position
    - goal_position: Position of the goal
    - holes_positions: All hole positions
    - safe_positions: All safe frozen lake positions

## look_ahead_holes tool
- After analyzing the grid, use this tool to look ahead and find holes around your position
- This tool takes the grid state and analyzes adjacent positions
- It will return:
    - action_list: Actions that would lead to holes (AVOID these)
    - holes_found: Specific hole positions found
    - safe_directions: Directions that are safe to move
- Use this information to avoid dangerous moves

## update_reward tool
- After analyzing the grid and holes, use this tool to determine the best action and its reward
- Consider the analysis from previous tools when choosing the best action
- The tool will return both the best action and its projected reward with reasoning

Your final goal is to reach the goal 'G' in the grid, and you must avoid the holes 'H' in the grid.
You must do this by maximizing the total reward for the agent.
"""
)

@agent.tool
async def analyze_grid(ctx: RunContext, grid_str: str) -> GridAnalysisResult:
    """Analyze the grid to find key positions and elements
    Args:
        grid_str: The current grid as a string
    Returns:
        GridAnalysisResult with agent position, goal position, holes, and safe positions
    """
    with logfire.span("analyze_grid", 
                     attributes={"grid_length": len(grid_str)}) as span:
        
        try:
            grid, agent_pos, goal_pos, holes = parse_grid(grid_str)
            
            # Find safe positions (F tiles)
            safe_positions = []
            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    if cell == 'F':
                        safe_positions.append(Position(row=row_idx, col=col_idx))
            
            result = GridAnalysisResult(
                agent_position=agent_pos or Position(row=0, col=0),
                goal_position=goal_pos or Position(row=0, col=0),
                holes_positions=holes,
                safe_positions=safe_positions
            )
            
            span.set_attribute("agent_position", f"({agent_pos.row}, {agent_pos.col})" if agent_pos else "None")
            span.set_attribute("goal_position", f"({goal_pos.row}, {goal_pos.col})" if goal_pos else "None")
            span.set_attribute("holes_count", len(holes))
            span.set_attribute("safe_positions_count", len(safe_positions))
            
            print(f"🔍 GRID_ANALYSIS: Agent at {agent_pos}, Goal at {goal_pos}, {len(holes)} holes found")
            
            return result
            
        except Exception as e:
            span.set_attribute("error", str(e))
            print(f"❌ GRID_ANALYSIS_ERROR: {e}")
            # Return a default result if parsing fails
            return GridAnalysisResult(
                agent_position=Position(row=0, col=0),
                goal_position=Position(row=0, col=0),
                holes_positions=[],
                safe_positions=[]
            )

@agent.tool
async def look_ahead_holes(ctx: RunContext, grid_str: str, thought_hole_direction: str) -> LookAheadHolesResult:
    """Look ahead to see if there are any holes in the grid around the agent's position
    Args:
        grid_str: The current grid as a string
        thought_hole_direction: The direction the agent should look ahead to see if there are any holes in the grid
    Returns:
        LookAheadHolesResult with actions to avoid, holes found, and safe directions
    """
    with logfire.span("look_ahead_holes", 
                     attributes={"thought_hole_direction": thought_hole_direction, 
                               "grid_length": len(grid_str)}) as span:
        
        try:
            grid, agent_pos, goal_pos, all_holes = parse_grid(grid_str)
            
            if not agent_pos:
                span.set_attribute("error", "Agent position not found")
                return LookAheadHolesResult(
                    action_list=[],
                    direction="Agent position not found",
                    thought_hole_direction=thought_hole_direction,
                    agent_position=Position(row=0, col=0),
                    holes_found=[],
                    safe_directions=[]
                )
            
            # Get adjacent positions
            adjacent = get_adjacent_positions(agent_pos, grid)
            
            # Analyze each direction
            dangerous_actions = []
            holes_found = []
            safe_directions = []
            
            for direction, adj_pos, content in adjacent:
                if content == 'H':
                    # This direction leads to a hole
                    if direction in ["left", "right", "up", "down"]:
                        dangerous_actions.append(Action(action=cast(Literal["left", "right", "up", "down"], direction)))
                    holes_found.append(adj_pos)
                elif content in ['F', 'G']:
                    # This direction is safe
                    safe_directions.append(direction)
            
            result = LookAheadHolesResult(
                action_list=dangerous_actions,
                direction=thought_hole_direction,
                thought_hole_direction=thought_hole_direction,
                agent_position=agent_pos,
                holes_found=holes_found,
                safe_directions=safe_directions
            )
            
            span.set_attribute("dangerous_actions", [a.action for a in dangerous_actions])
            span.set_attribute("holes_found_count", len(holes_found))
            span.set_attribute("safe_directions", safe_directions)
            
            print(f"🔍 LOOK_AHEAD: Agent at {agent_pos}, Dangerous actions: {[a.action for a in dangerous_actions]}, Safe: {safe_directions}")
            
            return result
            
        except Exception as e:
            span.set_attribute("error", str(e))
            print(f"❌ LOOK_AHEAD_ERROR: {e}")
            return LookAheadHolesResult(
                action_list=[],
                direction="Error in analysis",
                thought_hole_direction=thought_hole_direction,
                agent_position=Position(row=0, col=0),
                holes_found=[],
                safe_directions=[]
            )

@agent.tool
async def update_reward(ctx: RunContext, grid_str: str, predicted_best_action: Action, projected_reward: float) -> RewardUpdateResult:
    """Update the reward for the agent based on grid analysis
    Args:
        grid_str: The current grid as a string
        predicted_best_action: The best action the agent should take based on analysis
        projected_reward: The projected reward for the agent
    Returns:
        RewardUpdateResult containing the best action, its projected reward, and reasoning
    """
    with logfire.span("update_reward", 
                      attributes={"predicted_best_action": predicted_best_action.action, 
                                "projected_reward": projected_reward,
                                "grid_length": len(grid_str)}) as span:
        
        try:
            grid, agent_pos, goal_pos, holes = parse_grid(grid_str)
            
            if not agent_pos:
                reasoning = "Agent position not found in grid"
                span.set_attribute("reasoning", reasoning)
                return RewardUpdateResult(
                    best_action=predicted_best_action,
                    projected_reward=projected_reward,
                    reasoning=reasoning
                )
            
            # Calculate the new position if the action is taken
            new_pos = Position(row=agent_pos.row, col=agent_pos.col)
            if predicted_best_action.action == "up":
                new_pos.row -= 1
            elif predicted_best_action.action == "down":
                new_pos.row += 1
            elif predicted_best_action.action == "left":
                new_pos.col -= 1
            elif predicted_best_action.action == "right":
                new_pos.col += 1
            
            # Check if the new position is valid
            if (new_pos.row < 0 or new_pos.row >= len(grid) or 
                new_pos.col < 0 or new_pos.col >= len(grid[0])):
                reasoning = f"Action {predicted_best_action.action} would move out of bounds"
                span.set_attribute("reasoning", reasoning)
                return RewardUpdateResult(
                    best_action=predicted_best_action,
                    projected_reward=-1000,  # Heavy penalty for out of bounds
                    reasoning=reasoning
                )
            
            # Check what's at the new position
            new_content = grid[new_pos.row][new_pos.col]
            
            if new_content == 'H':
                reasoning = f"Action {predicted_best_action.action} leads to a hole - AVOID!"
                final_reward = -1000
            elif new_content == 'G':
                reasoning = f"Action {predicted_best_action.action} leads to the goal - EXCELLENT!"
                final_reward = 100
            elif new_content in ['F', 'S']:
                reasoning = f"Action {predicted_best_action.action} leads to safe frozen lake"
                final_reward = -1
            else:
                reasoning = f"Action {predicted_best_action.action} leads to unknown content: {new_content}"
                final_reward = projected_reward
            
            result = RewardUpdateResult(
                best_action=predicted_best_action,
                projected_reward=final_reward,
                reasoning=reasoning
            )
            
            span.set_attribute("new_position", f"({new_pos.row}, {new_pos.col})")
            span.set_attribute("new_content", new_content)
            span.set_attribute("final_reward", final_reward)
            span.set_attribute("reasoning", reasoning)
            
            print(f"💰 REWARD_UPDATE: Action {predicted_best_action.action} -> {new_content} at {new_pos}, Reward: {final_reward}")
            
            return result
            
        except Exception as e:
            span.set_attribute("error", str(e))
            reasoning = f"Error in reward calculation: {e}"
            print(f"❌ REWARD_UPDATE_ERROR: {e}")
            return RewardUpdateResult(
                best_action=predicted_best_action,
                projected_reward=projected_reward,
                reasoning=reasoning
            )

async def main():
    import re
    
    with logfire.span("frozen_lake_game_session") as session_span:
        state, info = env.reset()
        print(f"Starting state: {state}")
        conversation_messages = None
        step_count = 0
        total_reward = 0.0
        
        session_span.set_attribute("starting_state", state)
        
        while step_count < 30:  # Limit steps to prevent infinite loops
            with logfire.span("game_step", 
                            attributes={"step_number": step_count + 1, 
                                      "current_state": state}) as step_span:
                
                print(f"\nStep {step_count + 1}:")
                print(f"Current state: {state}")
                
                # Create game state for the agent, including the rendered grid
                rendered_grid_io = env.render()
                getvalue = getattr(rendered_grid_io, "getvalue", None)
                if callable(getvalue):
                    rendered_grid = str(getvalue())
                elif isinstance(rendered_grid_io, str):
                    rendered_grid = rendered_grid_io
                elif rendered_grid_io is None:
                    rendered_grid = ""
                else:
                    rendered_grid = str(rendered_grid_io)
                
                # --- Clean up ANSI codes and mark agent position as 'P' ---
                # Remove ANSI escape codes and replace the colored symbol with 'P'
                # The colored symbol is always the agent's current position
                # The ANSI code for red background is '\x1b[41m' and reset is '\x1b[0m'
                # Replace '\x1b[41m(.)\x1b[0m' with 'P'
                rendered_grid_clean = re.sub(r'\x1b\[41m(.)\x1b\[0m', 'P', rendered_grid)
                # Remove any other ANSI codes just in case
                rendered_grid_clean = re.sub(r'\x1b\[[0-9;]*m', '', rendered_grid_clean)
                
                # Debug: Print what the LLM agent sees
                print("Raw grid (with ANSI codes):")
                print(repr(rendered_grid))
                print("\nCleaned grid (what LLM agent sees):")
                print(rendered_grid_clean)
                print("\n" + "="*50)
                
                game_state = GameState(
                    state=state, 
                    action=None, 
                    reward=0, 
                    done=False, 
                    truncated=False,
                    rendered_grid=rendered_grid_clean,
                    total_reward=total_reward
                )
                
                if conversation_messages is None:
                    # Get action from agent     
                    result = await agent.run(f"Current game state: {game_state.model_dump()}")
                    print("First run - no conversation history")
                else:
                    # Get action from agent
                    result = await agent.run(f"Current game state: {game_state.model_dump()}", message_history=conversation_messages)
                    print(f"Using conversation history with {len(conversation_messages)} messages")
                
                agent_decision = result.output
                print(f"Agent chose: {agent_decision.action.action}")
                print(f"Projected reward: {agent_decision.projected_reward}")
                conversation_messages = result.all_messages()
                
                # Update total reward with projected reward from agent
                total_reward += agent_decision.projected_reward
                
                # Log agent decision
                step_span.set_attribute("agent_action", agent_decision.action.action)
                step_span.set_attribute("projected_reward", agent_decision.projected_reward)
                step_span.set_attribute("conversation_messages_count", len(conversation_messages))
                
                # Convert to gymnasium action and step
                gym_action = agent_decision.action.to_gym_action()
                print(f"Gymnasium action: {gym_action}")
                
                state, reward, done, truncated, info = env.step(gym_action)
                total_reward += float(reward)
                print(f"New state: {state}, Reward: {reward}, Done: {done}")
                print(f"Info: {info}")  # Log additional environment info
                
                # Log step results
                step_span.set_attribute("new_state", state)
                step_span.set_attribute("reward", reward)
                step_span.set_attribute("done", done)
                step_span.set_attribute("total_reward", total_reward)
                
                if done or truncated:
                    print(f"Game finished! Final reward: {total_reward}")
                    session_span.set_attribute("final_reward", total_reward)
                    session_span.set_attribute("total_steps", step_count + 1)
                    session_span.set_attribute("game_result", "completed" if done else "truncated")
                    break
                    
                step_count += 1
        
        env.close()
        print("Environment closed.")
    
if __name__ == "__main__":
    asyncio.run(main())