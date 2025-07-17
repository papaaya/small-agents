"""
Lets create a Sokoban agent that can solve the puzzle. Here is the prompt:

You are solving the Sokoban puzzle. You are the player and you need to push all boxes to
targets. When you are right next to a box, you can push it by moving in the same direction.
You cannot push a box through a wall, and you cannot pull a box. The answer should be a
sequence of actions, like <answer>Right || Right || Up</answer>.
The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player, S: player on target
Your available actions are:
Up, Down, Left, Right
You can make up to 10 actions, separated by the action separator " || "
Turn 1:
State:
######
######
#O####
#XP###
#__###
######
You have 10 actions left. Always output: <think> [Your thoughts] </think> <answer> [your
answer] </answer> with no extra text. Strictly follow this format. Max response length: 100
words (tokens).

Turn 2: ...
"""

"""
Sokoban Puzzle Solver Agent using Pydantic
"""

"""
Harder Sokoban Puzzle - Multiple boxes and targets (6x6 grid)
HARDER_PUZZLE = 
######
#____#
#_OXO_#
#_XPX_#
#_OXO_#
#____#
######

COMPLEX_PUZZLE = 
######
#____#
#_O_O_#
#_XPX_#
#_O_O_#
#____#
######

"""


from typing import List, Optional
from pydantic import BaseModel, Field
import re
from enum import Enum

import asyncio
from dataclasses import dataclass
from typing import Any, Tuple
from openai.types.shared import responses_model
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from devtools import debug
import logfire
import urllib.parse
from httpx import AsyncClient
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from openai.types.responses import WebSearchToolParam  
import dotenv

# Import the prompt manager
from prompt_manager import PromptManager

dotenv.load_dotenv()

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

model_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
model = OpenAIResponsesModel('o3-mini')

# Initialize prompt manager
prompt_manager = PromptManager()

def print_grid(grid: List[List[str]], title: str = "Sokoban Grid") -> None:
    """
    Pretty print the Sokoban grid
    
    Args:
        grid: 2D grid to print
        title: Optional title for the grid
    """
    print(f"\n{title}:")
    print("=" * (len(grid[0]) * 2 + 3))
    
    for i, row in enumerate(grid):
        print(f"{i:2d} | {' '.join(row)} |")
    
    print("=" * (len(grid[0]) * 2 + 3))
    print("    " + " ".join(f"{j:1d}" for j in range(len(grid[0]))))
    print()


class Action(str, Enum):
    """Available actions for the Sokoban agent"""
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"


class GameResult(str, Enum):
    """Result of the game"""
    WIN = "Win"
    LOSE = "Lose"

prompt_version = "v5"
system_prompt = prompt_manager.get_prompt(prompt_version)
prompt_info = prompt_manager.get_prompt_info(prompt_version)

print(f"🤖 Creating agent with prompt version: {prompt_version or 'default'}")
print(f"   Name: {prompt_info.get('name', 'Unknown')}")
print(f"   Description: {prompt_info.get('description', 'No description')}")

agent = Agent(model=model, 
            model_settings=model_settings,
            system_prompt=system_prompt,
            output_type=GameResult)


@agent.tool
async def valid_action(ctx: RunContext, action: Action) -> str:
    print(f"🔍 VALID_ACTION called with action: {action}")
    """Check if the action is valid"""
    print(f"Checking if action {action} is valid")
    valid_action_agent = Agent(model=model, 
                model_settings=model_settings,
                system_prompt="""
                You are checking if the action is valid.
                You are given the state of the puzzle and the action.
                You need to check if the action is valid. Below are some invalid actions:
                P does not move through walls (#).
                X does not move through walls (#).
                O does not move through walls (#).
                # does not move through walls (#).
                
                Below are some valid actions:
                P can move through empty spaces (_).
                X can move through empty spaces (_).
                O can move through empty spaces (_).
                # can move through empty spaces (_).
                P can move through targets (O).
                X can move through targets (O).
                When P is next to a box, P can push the box in the same direction as P moves.

                """)
    result = await valid_action_agent.run(action)
    return str(result)

@agent.tool
async def update_grid(ctx: RunContext, grid: List[List[str]], action: Action, player_position: List[int], box_position: List[int], target_position: List[int], think: str) -> Tuple[List[List[str]], str]:
    """
    Update the grid based on the valid action and return the new grid and the thought process in think variable.
    """
    print(f"🔄 UPDATE_GRID called with action: {action}")
    print("📋 Current grid state:")
    print_grid(grid, "Before Update")
    """Update the grid based on the valid action"""
    print(f"Updating grid based on action {action}")
    
    # Create a deep copy to avoid modifying the original grid
    new_grid = [row[:] for row in grid]
    
    if not player_position or len(player_position) != 2:
        error_msg = "Error: Invalid player position format"
        print(error_msg)
        return grid, error_msg
    
    player_row, player_col = player_position[0], player_position[1]
    
    # Calculate new player position based on action
    new_player_row, new_player_col = player_row, player_col
    
    if action == Action.UP:
        new_player_row = player_row - 1
    elif action == Action.DOWN:
        new_player_row = player_row + 1
    elif action == Action.LEFT:
        new_player_col = player_col - 1
    elif action == Action.RIGHT:
        new_player_col = player_col + 1
    
    # Edge case 1: Check bounds
    if (new_player_row < 0 or new_player_row >= len(new_grid) or 
        new_player_col < 0 or new_player_col >= len(new_grid[0])):
        error_msg = f"Error: Action {action} would move player out of bounds"
        print(error_msg)
        return grid, error_msg
    
    # Edge case 2: Check if new position is a wall
    if new_grid[new_player_row][new_player_col] == '#':
        error_msg = f"Error: Action {action} would move player into a wall"
        print(error_msg)
        return grid, error_msg
    
    # Edge case 3: Check if moving a box
    if new_grid[new_player_row][new_player_col] in ['X', '√']:
        # Calculate box's new position
        box_new_row = new_player_row + (new_player_row - player_row)
        box_new_col = new_player_col + (new_player_col - player_col)
        
        # Edge case 4: Check if box would move out of bounds
        if (box_new_row < 0 or box_new_row >= len(new_grid) or 
            box_new_col < 0 or box_new_col >= len(new_grid[0])):
            error_msg = f"Error: Action {action} would push box out of bounds"
            print(error_msg)
            return grid, error_msg
        
        # Edge case 5: Check if box would be pushed into a wall or another box
        if new_grid[box_new_row][box_new_col] in ['#', 'X', '√']:
            error_msg = f"Error: Action {action} would push box into obstacle"
            print(error_msg)
            return grid, error_msg
        
        # Move the box
        old_box_char = new_grid[new_player_row][new_player_col]
        # Determine new box character based on destination
        if new_grid[box_new_row][box_new_col] == 'O':
            new_box_char = '√'  # Box on target
        else:
            new_box_char = 'X'  # Box on empty space
        
        new_grid[box_new_row][box_new_col] = new_box_char
        new_grid[new_player_row][new_player_col] = '_'  # Clear box's old position
    
    # Move the player
    old_player_char = new_grid[player_row][player_col]
    
    # Determine new player character based on destination
    if new_grid[new_player_row][new_player_col] == 'O':
        new_player_char = 'S'  # Player on target
    else:
        new_player_char = 'P'  # Player on empty space
    
    new_grid[new_player_row][new_player_col] = new_player_char
    
    # Restore the old player position
    if old_player_char == 'S':
        new_grid[player_row][player_col] = 'O'  # Was player on target, restore target
    else:
        new_grid[player_row][player_col] = '_'  # Was player on empty space
    
    success_msg = f"Successfully moved player from ({player_row}, {player_col}) to ({new_player_row}, {new_player_col})"
    print(success_msg)
    print_grid(new_grid)
    return new_grid, think
    


@agent.tool
async def string_state_to_grid(ctx: RunContext, state_str: str) -> List[List[str]]:
    """Convert Sokoban string state to 2D grid (list of lists)"""
    lines = [line.strip() for line in state_str.strip().split('\n') if line.strip()]
    grid = [list(line) for line in lines]
    return grid


@agent.tool
async def find_player_position(ctx: RunContext, grid: List[List[str]]) -> tuple | None:
    """Find player position (P) in the grid"""
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'P':
                return (i, j)
    return None


@agent.tool
async def find_box_positions(ctx: RunContext, grid: List[List[str]]) -> List[tuple]:
    """Find all box positions (X) in the grid"""
    boxes = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'X':
                boxes.append((i, j))
    return boxes


@agent.tool
async def find_target_positions(ctx: RunContext, grid: List[List[str]]) -> List[tuple]:
    """Find all target positions (O) in the grid"""
    targets = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'O':
                targets.append((i, j))
    return targets


@agent.tool
async def grid_to_string_state(ctx: RunContext, grid: List[List[str]]) -> str:
    """Convert 2D grid back to Sokoban string state"""
    return '\n'.join([''.join(row) for row in grid])

@agent.tool
async def get_reward(ctx: RunContext, state: str, action: Action) -> float:
    print(f"💰 GET_REWARD called with action: {action}")
    """
    Get the reward for the action, given the state.
    Provides meaningful signals to guide the agent's learning.
    
    Args:
        state: Current game state as string
        action: Action taken
        
    Returns:
        Reward value (positive for good actions, negative for bad actions)
    """
    # Convert state string to grid for analysis
    lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
    grid = [list(line) for line in lines]
    
    reward = 0.0
    
    # Count current state metrics
    boxes_on_targets = sum(1 for row in grid for cell in row if cell == '√')
    total_boxes = sum(1 for row in grid for cell in row if cell in ['X', '√'])
    total_targets = sum(1 for row in grid for cell in row if cell in ['O', '√'])
    
    # Base reward for progress
    if boxes_on_targets > 0:
        reward += boxes_on_targets * 10.0  # +10 for each box on target
    
    # Check if puzzle is solved
    if boxes_on_targets == total_targets and total_targets > 0:
        reward += 100.0  # Big bonus for solving the puzzle
    
    # Penalty for invalid moves (would be caught by valid_action tool)
    # This is more of a safety net
    
    # Small penalty for each move to encourage efficiency
    reward -= 1.0
    
    # Bonus for moving towards targets
    # Find player position
    player_pos = None
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell in ['P', 'S']:
                player_pos = (i, j)
                break
        if player_pos:
            break
    
    if player_pos:
        # Check if player is near a box that can be pushed to target
        row, col = player_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check adjacent positions
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and 
                grid[new_row][new_col] == 'X'):
                # Check if pushing this box would move it towards a target
                box_new_row, box_new_col = new_row + dr, new_col + dc
                if (0 <= box_new_row < len(grid) and 0 <= box_new_col < len(grid[0]) and 
                    grid[box_new_row][box_new_col] == 'O'):
                    reward += 5.0  # Bonus for being in position to push box to target
    
    # Penalty for potential deadlocks
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'X':
                # Check if box is in corner
                if ((i == 0 or i == len(grid) - 1) and 
                    (j == 0 or j == len(grid[0]) - 1)):
                    reward -= 20.0  # Heavy penalty for corner deadlock
                
                # Check if box is against wall with no escape
                elif ((i == 0 or i == len(grid) - 1) and 
                      (j > 0 and j < len(grid[0]) - 1) and
                      grid[i][j-1] == '#' and grid[i][j+1] == '#'):
                    reward -= 15.0  # Penalty for wall deadlock
                elif ((j == 0 or j == len(grid[0]) - 1) and 
                      (i > 0 and i < len(grid) - 1) and
                      grid[i-1][j] == '#' and grid[i+1][j] == '#'):
                    reward -= 15.0  # Penalty for wall deadlock
    
    return reward

@agent.tool
async def verify_solution(ctx: RunContext) -> str:
    """Verify if the solution is correct"""
    verification_agent = Agent(model=model, 
                model_settings=model_settings,
                system_prompt="""
                You are verifying the solution to the Sokoban puzzle.
                You are given the state of the puzzle and the solution.
                You can find √  in the state. if all √ are in the initial target position, then the solution is correct.
                There could be multiple √ in the state, depending on the number of boxes.
                
                There is input state by the user and answer by the agent.
                You need to verify if the answer is correct.
                True or False.
                """)
    result = await verification_agent.run("True or False")
    print(f"Verification result: {result}")
    return str(result)

@agent.tool
async def restart_puzzle(ctx: RunContext, original_state: str) -> str:
    """
    Restart the puzzle from the original state
    
    Args:
        original_state: The original puzzle state to restart from
        
    Returns:
        Confirmation message
    """
    print("🔄 RESTARTING PUZZLE - Returning to original state")
    print("Original state:")
    print(original_state)
    return f"Puzzle restarted. Original state restored with {original_state.count('X')} boxes and {original_state.count('O')} targets."




async def solve_puzzle_with_prompt_version(puzzle_state: str, prompt_version: str = "v2", max_iterations: int = 50) -> dict:
    """
    Solve a Sokoban puzzle using a specific prompt version
    
    Args:
        puzzle_state: The initial puzzle state as a string
        prompt_version: Version of the prompt to use
        max_iterations: Maximum number of iterations to try
        
    Returns:
        Dictionary with results
    """
    print(f"🧩 Solving puzzle with prompt version: {prompt_version}")
    print("=" * 60)
    
    # Create agent with specific prompt version
    conversation_messages = None
    failure_count = 0
    restart_count = 0
    max_restarts = 3
    
    # Initialize current state
    current_state = puzzle_state
    print("📋 Initial State:")
    print_grid([list(line) for line in puzzle_state.strip().split('\n') if line.strip()], "Initial Grid")
    
    for iteration in range(max_iterations):
        try:
            print(f"\n🔄 Iteration {iteration + 1}")
            print("-" * 40)
            
            if conversation_messages is None:
                result = await agent.run(current_state)
            else:
                result = await agent.run("Continue solving the puzzle", message_history=conversation_messages)
            
            print(f"🤔 Agent thinking: {result.output}")
            
            # Calculate current reward manually
            lines = [line.strip() for line in current_state.strip().split('\n') if line.strip()]
            grid = [list(line) for line in lines]
            boxes_on_targets = sum(1 for row in grid for cell in row if cell == '√')
            total_targets = sum(1 for row in grid for cell in row if cell in ['O', '√'])
            
            # Simple reward calculation
            reward = boxes_on_targets * 10.0 - 1.0  # +10 per box on target, -1 for move
            if boxes_on_targets == total_targets and total_targets > 0:
                reward += 100.0  # Bonus for solving
            
            print(f"💰 Current reward: {reward}")
            print(f"📊 Progress: {boxes_on_targets}/{total_targets} boxes on targets")
            
            # Show current grid state
            print_grid(grid, f"Grid after iteration {iteration + 1}")
            
            if boxes_on_targets == total_targets and total_targets > 0:
                print("🎉 PUZZLE SOLVED!")
                print_grid(grid, "Final Solved Grid")
                break
            
            conversation_messages = result.all_messages()
            failure_count = 0  # Reset failure count on success
            
        except Exception as e:
            failure_count += 1
            print(f"❌ Error in iteration {iteration + 1}: {e}")
            
            # Check if we should restart
            if restart_count < max_restarts and failure_count >= 3:
                print(f"🔄 Attempting restart {restart_count + 1}/{max_restarts}")
                restart_count += 1
                conversation_messages = None  # Reset conversation
                failure_count = 0
                current_state = puzzle_state  # Reset to original state
                print("📋 Restarted to original state:")
                print_grid([list(line) for line in puzzle_state.strip().split('\n') if line.strip()], "Restarted Grid")
                continue
            
            # # Check if we should switch prompt version
            # new_version = prompt_manager.switch_prompt_version(prompt_version, failure_count)
            # if new_version != prompt_version:
            #     print(f"🔄 Switching to prompt version: {new_version}")
            #     prompt_version = new_version
            #     current_agent = create_agent(prompt_version)
            #     conversation_messages = None  # Reset conversation
            #     failure_count = 0
    
    return {
        "prompt_version_used": prompt_version,
        "iterations": iteration + 1,
        "restarts": restart_count,
        "final_result": result.output if 'result' in locals() else "Failed"
    }





async def main():
    conversation_messages = None

    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        state_solved_1 = """######
######
#O####
#XP###
#__###
######"""
        state = """######
#____#
#_O_O_#
#_XPX_#
#_____#
#____#
######"""

        print("🎮 Sokoban Agent Test")
        print("=" * 80)
        
        # Test the original solve function with grid updates
        print("Testing with grid visualization and reward tool...")
        result = await solve_puzzle_with_prompt_version(state, prompt_version, max_iterations=10)
        print(f"\n✅ Final result: {result}")


if __name__ == "__main__":
    asyncio.run(main())