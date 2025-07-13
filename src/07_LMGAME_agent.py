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
You can make up to 10 actions, separated by the action separator “ || ”
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
from typing import Any
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

dotenv.load_dotenv()

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel('gpt-4o')





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



class SokobanResponse(BaseModel):
    """Response model for the Sokoban agent"""
    think: str = Field(description="Agent's thinking process about the game state and strategy")
    answer: Action = Field(description="Only one of the following actions: Up, Down, Left, Right")

class GameResult(str, Enum):
    """Result of the game"""
    WIN = "Win"
    LOSE = "Lose"


agent = Agent(model=model, 
                model_settings=model_settings,
                system_prompt="""
                You are solving the Sokoban puzzle. You are the player and you need to push all boxes to
                targets. When you are right next to a box, you can push it by moving in the same direction.
                You cannot push a box through a wall, and you cannot pull a box. 

                When asked to continue the solving puzzle, you should look at the last updated state and the action you took.

                
                Symbols in the state:
                - P: player (you)
                - X: box (needs to be pushed to target)
                - O: target (where boxes should go)
                - √: box on target (solved)
                - S: player on target
                - #: wall (cannot move through)
                - _: empty space, Where P and X can move through    
                - √: box on target (solved) 
                
                Your available actions are EXACTLY: Up, Down, Left, Right
                You must select any one action.
                
                You can make up to 1000 actions
                Always output: <think> [Your thoughts about game and state] </think>
                Solving strategies:
                Simple:
                - Find the box that is closest to the target.
                - Push the box to the target.
                - Repeat until all boxes are on targets.
                - If you are stuck, try to find a new path.
                - If you are stuck, try to find a new path.
                - reflect on the action you took and the state you are in.
                - lets think step by step.
                

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

                You must use the verify_solution tool to verify your answer.
                You must use the valid_action tool to check if the action is valid.
                you must use the update_grid tool to update the grid based on the action.
                After updating the grid, you must use the verify_solution tool to verify if the solution is correct.
                if Not correct, you should proceed to the next action. and try to solve the puzzle.
            """,
            output_type=GameResult)


@agent.tool
async def valid_action(ctx: RunContext, action: Action) -> str:
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
async def update_grid(ctx: RunContext, grid: List[List[str]], action: Action, player_position: List[int], box_position: List[int], target_position: List[int]) -> List[List[str]] | str:
    """Update the grid based on the valid action"""
    print(f"Updating grid based on action {action}")
    
    # Create a deep copy to avoid modifying the original grid
    new_grid = [row[:] for row in grid]
    
    if not player_position or len(player_position) != 2:
        error_msg = "Error: Invalid player position format"
        print(error_msg)
        return error_msg
    
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
        return error_msg
    
    # Edge case 2: Check if new position is a wall
    if new_grid[new_player_row][new_player_col] == '#':
        error_msg = f"Error: Action {action} would move player into a wall"
        print(error_msg)
        return error_msg
    
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
            return error_msg
        
        # Edge case 5: Check if box would be pushed into a wall or another box
        if new_grid[box_new_row][box_new_col] in ['#', 'X', '√']:
            error_msg = f"Error: Action {action} would push box into obstacle"
            print(error_msg)
            return error_msg
        
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
    return new_grid
    


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
    """Get the reward for the action, given the state"""

    return -1.0

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

        print("Sending state to agent:")
        print(state)
        print("\nWaiting for agent response...")
        for i in range(100):
            try:
                if conversation_messages is None:
                    result = await agent.run(state)
                else:
                    result = await agent.run("lets continue the solving puzzle", message_history=conversation_messages)

                print("\nAgent Response:")
                print(result)                

                conversation_messages = result.all_messages()

                    
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())