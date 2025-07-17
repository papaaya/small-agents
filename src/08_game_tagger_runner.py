import asyncio
import random
from litellm.secret_managers.main import str_to_bool
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic_ai.models.openai import OpenAIModelSettings
from pydantic import BaseModel, Field
from typing import Literal, Tuple, List

# Configuration
MAX_MOVES = 16  # Configurable maximum number of moves

model = OpenAIResponsesModel('gpt-4o-mini')
model_settings = OpenAIModelSettings(temperature=0.3)

model_runner = OpenAIResponsesModel('gpt-4o-mini')
model_settings_runner = OpenAIModelSettings(temperature=0.3)

system_prompt_tagger = """
You are a game tagger. Your objective is to catch the runner in a 2D grid provided to you.
You need to provide the best action to move the tagger towards the runner position.

Available tools:
1. tagger_move - Move the tagger to a new position

Reward system:
- -1 for each move
- +100 if you catch the runner
- Your goal is to maximize your total reward

Strategy tips:
- Consider the distance to runner when choosing moves
- Balance between aggressive pursuit and efficient movement
- Monitor your current reward and the runner's reward
- Beware of Boundary conditions of the grid. If you are at the boundary, you cannot move out of the grid or hit the wall.
- The game ends after 16 moves if runner is not caught
"""

system_prompt_runner = """
You are a game runner. You are given a 2D game state. 
You need to provide the best action to move the runner away from the tagger position.

Available tools:
1. runner_move - Move the runner to a new position

Reward system:
- +1 for each move
- +100 if you survive 16 moves
- -200 if you get caught
- Your goal is to maximize your total reward

Strategy tips:
- Keep distance from the tagger
- Monitor moves remaining (16 total)
- Balance between survival and reward accumulation
- Consider the tagger's current reward and strategy
- Beware of Boundary conditions of the grid. If you are at the boundary, you cannot move out of the grid or hit the wall.
"""

class GameResult(BaseModel):
    game_result: Literal["win", "lose", "incomplete"]

class Action(BaseModel):
    action: Literal["up", "down", "left", "right"]

class Position(BaseModel):
    row: int = Field(description="Row position (0-2)")
    col: int = Field(description="Column position (0-2)")

class GridState(BaseModel):
    grid: List[List[str]] = Field(description="The 3x3 game grid")
    runner_position: Position = Field(description="Current runner position")
    tagger_position: Position = Field(description="Current tagger position")

class GameStats(BaseModel):
    tagger_reward: int = Field(description="Current tagger reward", default=0)
    runner_reward: int = Field(description="Current runner reward", default=0)
    move_count: int = Field(description="Number of moves made", default=0)
    game_over: bool = Field(description="Whether the game is over", default=False)
    winner: str = Field(description="Winner of the game", default="")

agent_tagger = Agent(model=model, 
            model_settings=model_settings,
            system_prompt=system_prompt_tagger,
            output_type=Action)

agent_runner = Agent(model=model_runner, 
            model_settings=model_settings_runner,
            system_prompt=system_prompt_runner,
            output_type=Action)


def create_grid_2d(runner_position: Tuple[int, int], tagger_position: Tuple[int, int]) -> List[List[str]]:
    """
    create a 2d grid with empty cells, runner and tagger positions.
    "R" is the runner position.
    "T" is the tagger position.
    " " is an empty cell.
    """
    grid = [
        [" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "]
    ]
    #place runner and tagger randomly
    runner_position = runner_position
    tagger_position = tagger_position
    grid[runner_position[0]][runner_position[1]] = "R"
    grid[tagger_position[0]][tagger_position[1]] = "T"
    print(grid)
    return grid

@agent_tagger.tool
def tagger_move(ctx: RunContext, grid_state: GridState, action: str) -> GridState:
    """
    Move the tagger to the new position.
    Update the grid with the new tagger position.
    """
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    # Convert Position to tuple for calculation
    tagger_pos = (grid_state.tagger_position.row, grid_state.tagger_position.col)
    new_tagger_position = (tagger_pos[0] + move_action_map[action][0], tagger_pos[1] + move_action_map[action][1])
    
    # Check boundary conditions
    if (0 <= new_tagger_position[0] < len(grid_state.grid) and 0 <= new_tagger_position[1] < len(grid_state.grid[0])):
        # Create new grid
        new_grid = [row[:] for row in grid_state.grid]  # Deep copy
        new_grid[tagger_pos[0]][tagger_pos[1]] = " "
        new_grid[new_tagger_position[0]][new_tagger_position[1]] = "T"
        
        return GridState(
            grid=new_grid,
            runner_position=grid_state.runner_position,
            tagger_position=Position(row=new_tagger_position[0], col=new_tagger_position[1])
        )
    else:
        # Invalid move - return same state
        return grid_state

@agent_runner.tool
def runner_move(ctx: RunContext, grid_state: GridState, action: str) -> GridState:
    """
    Move the runner to the new position.
    Update the grid with the new runner position.
    """
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    # Convert Position to tuple for calculation
    runner_pos = (grid_state.runner_position.row, grid_state.runner_position.col)
    new_runner_position = (runner_pos[0] + move_action_map[action][0], runner_pos[1] + move_action_map[action][1])
    
    # Check boundary conditions
    if (0 <= new_runner_position[0] < len(grid_state.grid) and 0 <= new_runner_position[1] < len(grid_state.grid[0])):
        # Create new grid
        new_grid = [row[:] for row in grid_state.grid]  # Deep copy
        new_grid[runner_pos[0]][runner_pos[1]] = " "
        new_grid[new_runner_position[0]][new_runner_position[1]] = "R"
        
        return GridState(
            grid=new_grid,
            runner_position=Position(row=new_runner_position[0], col=new_runner_position[1]),
            tagger_position=grid_state.tagger_position
        )
    else:
        # Invalid move - return same state
        return grid_state

# Helper function to calculate tagger reward without RunContext (for main function)
def calculate_tagger_reward_helper(grid_state: GridState, game_stats: GameStats, action: Action) -> int:
    """
    Calculate reward for tagger's move.
    -1 for each move
    +100 if tagger catches runner (positions are the same)
    Returns the total reward for this move.
    """
    # Base reward: -1 for moving
    reward = -1
    
    # Check if tagger caught runner
    if (grid_state.tagger_position.row == grid_state.runner_position.row and 
        grid_state.tagger_position.col == grid_state.runner_position.col):
        reward += 100  # Bonus for catching runner
    
    return reward

# Helper function to calculate runner reward without RunContext (for main function)
def calculate_runner_reward_helper(grid_state: GridState, game_stats: GameStats, action: Action) -> int:
    """
    Calculate reward for runner's move.
    +1 for each move
    +100 if runner survives 100 moves
    -200 if runner gets caught
    Returns the total reward for this move.
    """
    # Base reward: +1 for moving
    reward = 1
    
    # Check if runner got caught
    if (grid_state.tagger_position.row == grid_state.runner_position.row and 
        grid_state.tagger_position.col == grid_state.runner_position.col):
        reward -= 200  # Penalty for getting caught
    
    # Check if runner survived 100 moves
    if game_stats.move_count >= MAX_MOVES:
        reward += 100  # Bonus for surviving
    
    return reward

# Helper function to update grid state manually
def update_grid_state_after_tagger_move(grid_state: GridState, action: Action) -> GridState:
    """Helper function to update grid state after tagger move"""
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    tagger_pos = (grid_state.tagger_position.row, grid_state.tagger_position.col)
    new_tagger_position = (tagger_pos[0] + move_action_map[action.action][0], tagger_pos[1] + move_action_map[action.action][1])
    
    if (0 <= new_tagger_position[0] < 3 and 0 <= new_tagger_position[1] < 3):
        new_grid = [row[:] for row in grid_state.grid]
        new_grid[tagger_pos[0]][tagger_pos[1]] = " "
        new_grid[new_tagger_position[0]][new_tagger_position[1]] = "T"
        
        return GridState(
            grid=new_grid,
            runner_position=grid_state.runner_position,
            tagger_position=Position(row=new_tagger_position[0], col=new_tagger_position[1])
        )
    else:
        return grid_state

def update_grid_state_after_runner_move(grid_state: GridState, action: Action) -> GridState:
    """Helper function to update grid state after runner move"""
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    runner_pos = (grid_state.runner_position.row, grid_state.runner_position.col)
    new_runner_position = (runner_pos[0] + move_action_map[action.action][0], runner_pos[1] + move_action_map[action.action][1])
    
    if (0 <= new_runner_position[0] < 3 and 0 <= new_runner_position[1] < 3):
        new_grid = [row[:] for row in grid_state.grid]
        new_grid[runner_pos[0]][runner_pos[1]] = " "
        new_grid[new_runner_position[0]][new_runner_position[1]] = "R"
        
        return GridState(
            grid=new_grid,
            runner_position=Position(row=new_runner_position[0], col=new_runner_position[1]),
            tagger_position=grid_state.tagger_position
        )
    else:
        return grid_state

def check_terminal_condition(grid: List[List[str]], runner_position: Tuple[int, int], tagger_position: Tuple[int, int], number_of_moves: int) -> Tuple[bool, int]:
    """
    Check if the game is terminal. And list the number of moves and winner.
    If the game is terminal, return True, number_of_moves, winner.
    If the game is not terminal, return False, number_of_moves, winner.
    """
    if runner_position == tagger_position:
        print("Tagger caught the runner")
        return True, number_of_moves
    else: 
        if number_of_moves >= MAX_MOVES:
            print("Runner escaped")
            return True, number_of_moves
        else:
            print("Game is not terminal")
            return False, number_of_moves

def create_10x10_grid(runner_position: Position, tagger_position: Position) -> List[List[str]]:
    """
    Create a 10x10 grid with runner and tagger positions.
    """
    grid = [[" " for _ in range(10)] for _ in range(10)]
    grid[runner_position.row][runner_position.col] = "R"
    grid[tagger_position.row][tagger_position.col] = "T"
    return grid

def update_grid_state_after_tagger_move_10x10(grid_state: GridState, action: Action) -> GridState:
    """Helper function to update grid state after tagger move on 10x10 grid"""
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    tagger_pos = (grid_state.tagger_position.row, grid_state.tagger_position.col)
    new_tagger_position = (tagger_pos[0] + move_action_map[action.action][0], tagger_pos[1] + move_action_map[action.action][1])
    
    if (0 <= new_tagger_position[0] < 10 and 0 <= new_tagger_position[1] < 10):
        new_grid = [row[:] for row in grid_state.grid]
        new_grid[tagger_pos[0]][tagger_pos[1]] = " "
        new_grid[new_tagger_position[0]][new_tagger_position[1]] = "T"
        
        return GridState(
            grid=new_grid,
            runner_position=grid_state.runner_position,
            tagger_position=Position(row=new_tagger_position[0], col=new_tagger_position[1])
        )
    else:
        return grid_state

def update_grid_state_after_runner_move_10x10(grid_state: GridState, action: Action) -> GridState:
    """Helper function to update grid state after runner move on 10x10 grid"""
    move_action_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    
    runner_pos = (grid_state.runner_position.row, grid_state.runner_position.col)
    new_runner_position = (runner_pos[0] + move_action_map[action.action][0], runner_pos[1] + move_action_map[action.action][1])
    
    if (0 <= new_runner_position[0] < 10 and 0 <= new_runner_position[1] < 10):
        new_grid = [row[:] for row in grid_state.grid]
        new_grid[runner_pos[0]][runner_pos[1]] = " "
        new_grid[new_runner_position[0]][new_runner_position[1]] = "R"
        
        return GridState(
            grid=new_grid,
            runner_position=Position(row=new_runner_position[0], col=new_runner_position[1]),
            tagger_position=grid_state.tagger_position
        )
    else:
        return grid_state

async def play_10x10_game(grid_state: GridState, game_stats: GameStats, conversation_messages_tagger, conversation_messages_runner):
    """
    Continue the game on a 10x10 grid after the 3x3 game ends.
    """
    print("\n🎮 Starting 10x10 Grid Game!")
    print("📏 Grid size: 10x10")
    print("🎯 Same reward system applies")
    
    # Reset game stats for 10x10 game
    game_stats.move_count = 0
    game_stats.game_over = False
    game_stats.winner = ""
    
    # Initialize 10x10 grid with current positions
    runner_position = Position(row=0, col=0)
    tagger_position = Position(row=9, col=9)
    grid_10x10 = create_10x10_grid(runner_position, tagger_position)
    
    grid_state = GridState(
        grid=grid_10x10,
        runner_position=runner_position,
        tagger_position=tagger_position
    )
    
    print(f"Initial positions - Runner: ({runner_position.row}, {runner_position.col}), Tagger: ({tagger_position.row}, {tagger_position.col})")
    
    while not game_stats.game_over:
        # Tagger's turn
        print(f"\n🔄 Move {game_stats.move_count + 1}: Tagger's turn (10x10)")
        print("Current grid (10x10):")
        for i, row in enumerate(grid_state.grid):
            print(f"{i:2d}: {row}")
        print(f"Current rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")
            
        result = await agent_tagger.run(f"""Continue the game on a 10x10 grid. Catch the runner!
Current state:
- Grid size: 10x10
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.tagger_reward}
- Move count: {game_stats.move_count}

Choose your action: up, down, left, or right to catch the runner!
Remember: You get -1 for each move, but +100 if you catch the runner.""", message_history=conversation_messages_tagger)
        
        conversation_messages_tagger = result.all_messages()
        print(f"Tagger action: {result.output.action}")
        
        # Update grid state after tagger move
        grid_state = update_grid_state_after_tagger_move_10x10(grid_state, result.output)
        
        # Calculate tagger reward
        tagger_reward = calculate_tagger_reward_helper(grid_state, game_stats, result.output)
        game_stats.tagger_reward += tagger_reward
        print(f"Tagger reward for this move: {tagger_reward}")
        
        # Check if tagger caught runner
        if grid_state.runner_position.row == grid_state.tagger_position.row and grid_state.runner_position.col == grid_state.tagger_position.col:
            game_stats.game_over = True
            game_stats.winner = "Tagger"
            print("🏁 Game Over! Tagger wins on 10x10 grid!")
            break

        # Runner's turn
        print(f"\n🔄 Move {game_stats.move_count + 1}: Runner's turn (10x10)")
        print("Current grid (10x10):")
        for i, row in enumerate(grid_state.grid):
            print(f"{i:2d}: {row}")
        print(f"Current rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")
            
        result = await agent_runner.run(f"""Continue escaping on a 10x10 grid. Stay alive!
Current state:
- Grid size: 10x10
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.runner_reward}
- Move count: {game_stats.move_count}

Choose your action: up, down, left, or right to escape the tagger!
Remember: You get +1 for each move, +100 for surviving {MAX_MOVES} moves, but -200 if caught.""", message_history=conversation_messages_runner)
        
        conversation_messages_runner = result.all_messages()
        print(f"Runner action: {result.output.action}")
        
        # Update grid state after runner move
        grid_state = update_grid_state_after_runner_move_10x10(grid_state, result.output)
        
        game_stats.move_count += 1
        
        # Calculate runner reward
        runner_reward = calculate_runner_reward_helper(grid_state, game_stats, result.output)
        game_stats.runner_reward += runner_reward
        print(f"Runner reward for this move: {runner_reward}")
        
        # Check if runner escaped (100 moves limit) or tagger caught runner
        if game_stats.move_count >= MAX_MOVES:
            game_stats.game_over = True
            game_stats.winner = "Runner"
            print(f"🏁 Game Over! Runner wins by surviving {MAX_MOVES} moves on 10x10 grid!")
            break
        
        if grid_state.runner_position.row == grid_state.tagger_position.row and grid_state.runner_position.col == grid_state.tagger_position.col:
            game_stats.game_over = True
            game_stats.winner = "Tagger"
            print("🏁 Game Over! Tagger wins on 10x10 grid!")
            break

    print(f"\n📊 10x10 Game Summary:")
    print(f"Total moves: {game_stats.move_count}")
    print(f"Winner: {game_stats.winner}")
    print(f"Final positions - Runner: ({grid_state.runner_position.row}, {grid_state.runner_position.col}), Tagger: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})")
    print(f"Final rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")

async def main():
    
    conversation_messages_tagger = None
    conversation_messages_runner = None
    
    # Initialize game state
    runner_position = Position(row=0, col=0)
    tagger_position = Position(row=2, col=2)
    grid = [
        ["R", " ", " "],
        [" ", " ", " "],
        [" ", " ", "T"]
    ]
    
    grid_state = GridState(
        grid=grid,
        runner_position=runner_position,
        tagger_position=tagger_position
    )
    
    game_stats = GameStats()
    
    print("🎮 Starting Tag Game with Reward System!")
    print(f"Initial positions - Runner: ({runner_position.row}, {runner_position.col}), Tagger: ({tagger_position.row}, {tagger_position.col})")
    print("📊 Reward System:")
    print("   Tagger: -1 per move, +100 for catching runner")
    print(f"   Runner: +1 per move, +100 for surviving {MAX_MOVES} moves, -200 if caught")
    print(f"🎯 Game ends after {MAX_MOVES} moves if runner is not caught")
    
    while not game_stats.game_over:
        # Tagger's turn
        print(f"\n🔄 Move {game_stats.move_count + 1}: Tagger's turn")
        print("Current grid:")
        for row in grid_state.grid:
            print(row)
        print(f"Current rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")
            
        if conversation_messages_tagger is None:
            result = await agent_tagger.run(f"""Analyze the given grid and decide the best action to catch the runner. 
Current state:
- Grid: {grid_state.grid}
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.tagger_reward}
- Runner's current reward: {game_stats.runner_reward}
- Move count: {game_stats.move_count}
- Moves remaining: {MAX_MOVES - game_stats.move_count}

Use your tools to analyze the situation and make the best strategic move!
Remember: You get -1 for each move, but +100 if you catch the runner. Choose wisely!""")
        else:
            result = await agent_tagger.run(f"""Continue trying to catch the runner. 
Current state:
- Grid: {grid_state.grid}
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.tagger_reward}
- Runner's current reward: {game_stats.runner_reward}
- Move count: {game_stats.move_count}
- Moves remaining: {MAX_MOVES - game_stats.move_count}

Use your tools to analyze the situation and make the best strategic move!
Remember: You get -1 for each move, but +100 if you catch the runner. Choose wisely!""", message_history=conversation_messages_tagger)
        
        conversation_messages_tagger = result.all_messages()
        print(f"Tagger action: {result.output.action}")
        
        # Update grid state after tagger move
        grid_state = update_grid_state_after_tagger_move(grid_state, result.output)
        
        # Calculate tagger reward
        tagger_reward = calculate_tagger_reward_helper(grid_state, game_stats, result.output)
        game_stats.tagger_reward += tagger_reward
        print(f"Tagger reward for this move: {tagger_reward}")
        
        # Check if tagger caught runner
        if grid_state.runner_position.row == grid_state.tagger_position.row and grid_state.runner_position.col == grid_state.tagger_position.col:
            game_stats.game_over = True
            game_stats.winner = "Tagger"
            print("🏁 Game Over! Tagger wins!")
            break

        # Runner's turn
        print(f"\n🔄 Move {game_stats.move_count + 1}: Runner's turn")
        print("Current grid:")
        for row in grid_state.grid:
            print(row)
        print(f"Current rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")
            
        if conversation_messages_runner is None:
            result = await agent_runner.run(f"""Move the runner to escape from the tagger. 
Current state:
- Grid: {grid_state.grid}
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.runner_reward}
- Tagger's current reward: {game_stats.tagger_reward}
- Move count: {game_stats.move_count}
- Moves remaining: {MAX_MOVES - game_stats.move_count}

Use your tools to analyze the situation and make the best strategic move!
Remember: You get +1 for each move, +100 for surviving 100 moves, but -200 if caught. Stay alive!""")
        else:
            result = await agent_runner.run(f"""Continue escaping from the tagger. 
Current state:
- Grid: {grid_state.grid}
- Runner position: ({grid_state.runner_position.row}, {grid_state.runner_position.col})
- Tagger position: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})
- Your current reward: {game_stats.runner_reward}
- Tagger's current reward: {game_stats.tagger_reward}
- Move count: {game_stats.move_count}
- Moves remaining: {MAX_MOVES - game_stats.move_count}

Use your tools to analyze the situation and make the best strategic move!
Remember: You get +1 for each move, +100 for surviving 100 moves, but -200 if caught. Stay alive!""", message_history=conversation_messages_runner)
        
        conversation_messages_runner = result.all_messages()
        print(f"Runner action: {result.output.action}")
        
        # Update grid state after runner move
        grid_state = update_grid_state_after_runner_move(grid_state, result.output)
        
        game_stats.move_count += 1
        
        # Calculate runner reward
        runner_reward = calculate_runner_reward_helper(grid_state, game_stats, result.output)
        game_stats.runner_reward += runner_reward
        print(f"Runner reward for this move: {runner_reward}")
        
        # Check if runner escaped (100 moves limit) or tagger caught runner
        if game_stats.move_count >= MAX_MOVES:
            game_stats.game_over = True
            game_stats.winner = "Runner"
            print("🏁 Game Over! Runner wins by surviving 100 moves!")
            break
        
        if grid_state.runner_position.row == grid_state.tagger_position.row and grid_state.runner_position.col == grid_state.tagger_position.col:
            game_stats.game_over = True
            game_stats.winner = "Tagger"
            print("🏁 Game Over! Tagger wins!")
            break

    print(f"\n📊 Final Game Summary:")
    print(f"Total moves: {game_stats.move_count}")
    print(f"Winner: {game_stats.winner}")
    print(f"Final positions - Runner: ({grid_state.runner_position.row}, {grid_state.runner_position.col}), Tagger: ({grid_state.tagger_position.row}, {grid_state.tagger_position.col})")
    print(f"Final rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")

    # Continue with 10x10 game
    print("\n" + "="*60)
    print("🎮 CONTINUING TO 10x10 GRID GAME")
    print("="*60)
    
    await play_10x10_game(grid_state, game_stats, conversation_messages_tagger, conversation_messages_runner)
    
    print("\n" + "="*60)
    print("🏁 COMPLETE GAME SESSION FINISHED")
    print("="*60)
    print(f"🎯 Combined rewards - Tagger: {game_stats.tagger_reward}, Runner: {game_stats.runner_reward}")
    print("🎮 Both 3x3 and 10x10 games completed!")

if __name__ == "__main__":
    asyncio.run(main())