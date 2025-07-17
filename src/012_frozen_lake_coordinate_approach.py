from pydantic_ai import Agent, Tool, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from typing import List, Literal
import asyncio
import json
from dataclasses import dataclass

import logfire
import dotenv

dotenv.load_dotenv()

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

class Actions(BaseModel):
    action: Literal["up", "down", "left", "right"]


class CurrentState(BaseModel):
    x: int = Field(default=0)
    y: int = Field(default=0)

class OneAction(BaseModel):
    action: Actions
    reward: float = Field(default=0)
    current_state: CurrentState
    terminal_state: bool = Field(default=False)
    reason: str = Field(description="Reason for the terminal state")
    selected_actions: List[str] = Field(default_factory=list, description="List of actions taken so far")
    path_taken: List[CurrentState] = Field(default_factory=list, description="List of positions visited so far")
    

class ActionList(BaseModel):
    actions: List[Actions]
    current_state: CurrentState
    reward: float = Field(default=0)

class GridDict(BaseModel):
    position_key: str
    position_value: List[CurrentState]

class GridJsonList(BaseModel):
    grid_dict: List[GridDict]

class TerminalStateResponse(BaseModel):
    """Response for terminal state checking"""
    is_terminal: bool = Field(description="Whether the game has ended")
    terminal_type: str = Field(description="Type of terminal state: 'win', 'loss', or 'none'")
    message: str = Field(description="Description of what happened")
    position: CurrentState = Field(description="Current position of the agent")

@dataclass
class GameDeps:
    """Dependencies for the frozen lake game"""
    grid_analysis: GridJsonList | None = None

def pretty_print_grid_data(grid_data: GridJsonList, original_grid: str = ""):
    """Pretty print the grid data in a readable format"""
    print("=" * 60)
    print("GRID ANALYSIS RESULTS")
    print("=" * 60)
    
    if original_grid:
        print("\n📋 ORIGINAL GRID:")
        print("-" * 30)
        print(original_grid.strip())
        print("-" * 30)
    
    print("\n🗺️  GRID COORDINATES BY CHARACTER:")
    print("-" * 40)
    
    for grid_item in grid_data.grid_dict:
        char = grid_item.position_key
        positions = grid_item.position_value
        
        # Format positions nicely
        pos_str = ", ".join([f"({pos.x}, {pos.y})" for pos in positions])
        
        # Add emoji and description based on character
        if char == "S":
            print(f"🚀 START (S): {pos_str}")
        elif char == "G":
            print(f"🎯 GOAL (G): {pos_str}")
        elif char == "H":
            print(f"🕳️  HOLE (H): {pos_str}")
        elif char == "F":
            print(f"❄️  FROZEN (F): {pos_str}")
        else:
            print(f"❓ UNKNOWN ({char}): {pos_str}")
    
    print("\n📊 SUMMARY:")
    print("-" * 20)
    total_positions = sum(len(item.position_value) for item in grid_data.grid_dict)
    print(f"Total positions analyzed: {total_positions}")
    
    for grid_item in grid_data.grid_dict:
        char = grid_item.position_key
        count = len(grid_item.position_value)
        print(f"'{char}': {count} position(s)")
    
    print("=" * 60)


move_agent = Agent(
    "openai:gpt-4o",
    output_type=OneAction,
    deps_type=GameDeps,
    system_prompt="""
    You are expert at path planning and finding goal position in a grid. You understand the grid and the characters in the grid.
    Use run_grid_analysis tool to get the grid data.
    IMPORTANT: Always use validate_move tool to check if a move is possible before making it.
    You are given a rectangular grid of characters representing a Frozen Lake environment. 

    
    # Symbols:
    • S - starting position of the player  
    • P - the player, starts from position S. 
    • F - frozen ice (safe to walk on)  
    • H - holes (stepping here kills the player)  
    • G - goal (the player must reach here)

    # Game rules:
    1. The player P can move up, down, left, or right.
    2. The player P can only move if the cell is not a hole. If the player steps on a hole, the game ends. you must return -1 reward.
    3. The player P starts from position S.
    4. you should continue to explore the paths for P to reach the goal G.
    5. Reward calculation:
        - If the player is at the goal, the reward should be 1.
        - If the player is killed by a hole, before reaching the goal, the reward should be -1. Game ends.
        - If the player is not at the goal, the reward should be 0.

    # Tools:
    - validate_move: Use this to check if a move from current position to a new position is valid
    - check_terminal_state: Use this to check if the current position is terminal (win/loss)
    
    Both tools return a structured response with:
    - is_terminal: boolean indicating if the game has ended
    - terminal_type: "win", "loss", or "none"
    - message: description of what happened
    - position: current position of the agent

    # Strategies and tips:
    1. Parse the grid into a 2D coordinate system, using 1-based indexing: (row, column), where (1,1) is the top-left cell.  
    2. Think step by step and plan a safe path from S to G that never steps onto H.  
    3. Write out each move as one of: Up, Down, Left, Right.  
    4. Be aware of deadends, try to avoid them. In some cases task could be impossible to solve.
    5. ALWAYS use validate_move tool before making any move to ensure it's valid.
    6. Set the reward based on the terminal state:
        - If terminal_type is "win": reward = 1, terminal_state = true
        - If terminal_type is "loss": reward = -1, terminal_state = true  
        - If terminal_type is "none": reward = 0, terminal_state = false
        
    Example output:

    ```json
    {
    "action": {"action": "down"},
    "reward": 0,
    "current_state": {"x": 2, "y": 1},
    "terminal_state": false
    }
    ```
    """
)



player_agent = Agent(
    "openai:gpt-4o", 
    output_type=GridJsonList,
    system_prompt="""
    You are expert at path planning and findind goal position in a grid. You understand the grid and the characters in the grid.
    You are given a grid of characters in plain text, for example:

    SFFH
    FFFF
    FHFH
    FGFH
    

    1. Parses the input into rows and columns.
    2. Assigns coordinates to each cell—by default using 1-based indexing with (row, column), where (1, 1) is the top-left.
    3. Outputs a JSON object that maps each distinct character to a list of its coordinates.

    • If there are multiple occurrences of the same character, list them all.
    • Format coordinates as (row, column).
    • If asked, also provide a zero-based version by subtracting 1 from each index.

    Example output for the sample grid above:

    ```json
    {
    "S": [(1, 1)],
    "F": [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 3), (4, 1), (4, 3)],
    "H": [(1, 4), (3, 2), (3, 4), (4, 4)],
    "G": [(4, 2)]
    }
        """
)


@move_agent.tool
async def run_grid_analysis(ctx: RunContext[GameDeps], grid: str) -> GridJsonList:
    with logfire.span("Running player agent"):
        result = await player_agent.run(grid)
        pretty_print_grid_data(result.output, grid)
        # Store in deps
        ctx.deps.grid_analysis = result.output
        return result.output

@move_agent.tool
async def validate_move(ctx: RunContext[GameDeps], from_position: CurrentState, reason_for_action: str, action: str) -> TerminalStateResponse:
    """Validate if a move is possible and return the resulting position"""
    with logfire.span("Validating move"):
        try:
            grid_analysis = ctx.deps.grid_analysis
            if not grid_analysis:
                return TerminalStateResponse(
                    is_terminal=False,
                    terminal_type="none",
                    message="No grid analysis available",
                    position=from_position
                )
            
            # Calculate new position based on action
            new_x, new_y = from_position.x, from_position.y
            if action == "up":
                new_x = max(1, from_position.x - 1)
            elif action == "down":
                new_x = min(4, from_position.x + 1)
            elif action == "left":
                new_y = max(1, from_position.y - 1)
            elif action == "right":
                new_y = min(4, from_position.y + 1)
            
            new_position = CurrentState(x=new_x, y=new_y)
            
            print(f"🔄 MOVE VALIDATION: {action} from ({from_position.x}, {from_position.y}) to ({new_x}, {new_y})")
            print(f"Reason for action: {reason_for_action}")
            # Check if new position is valid (within bounds)
            if new_x < 1 or new_x > 4 or new_y < 1 or new_y > 4:
                print(f"❌ INVALID MOVE: Position ({new_x}, {new_y}) is out of bounds")
                return TerminalStateResponse(
                    is_terminal=True,
                    terminal_type="loss",
                    message=f"Invalid move: Position ({new_x}, {new_y}) is out of bounds",
                    position=from_position
                )
            
            # Find goal and hole positions
            goal_positions = []
            hole_positions = []
            
            for grid_item in grid_analysis.grid_dict:
                if grid_item.position_key == 'G':
                    goal_positions = grid_item.position_value
                elif grid_item.position_key == 'H':
                    hole_positions = grid_item.position_value
            
            # Check if new position is a hole
            for hole_pos in hole_positions:
                if new_x == hole_pos.x and new_y == hole_pos.y:
                    print(f"💀 INVALID MOVE: Position ({new_x}, {new_y}) is a hole!")
                    return TerminalStateResponse(
                        is_terminal=True,
                        terminal_type="loss",
                        message=f"Agent fell into hole at position ({new_x}, {new_y})",
                        position=new_position
                    )
            
            # Check if new position is the goal
            for goal_pos in goal_positions:
                if new_x == goal_pos.x and new_y == goal_pos.y:
                    print(f"🎉 VALID MOVE: Position ({new_x}, {new_y}) is the goal!")
                    return TerminalStateResponse(
                        is_terminal=True,
                        terminal_type="win",
                        message=f"Agent reached goal at position ({new_x}, {new_y})",
                        position=new_position
                    )
            
            print(f"✅ VALID MOVE: Position ({new_x}, {new_y}) is safe")
            return TerminalStateResponse(
                is_terminal=False,
                terminal_type="none",
                message=f"Valid move to safe position ({new_x}, {new_y})",
                position=new_position
            )
            
        except Exception as e:
            print(f"❌ Error in validate_move: {e}")
            return TerminalStateResponse(
                is_terminal=False,
                terminal_type="none",
                message=f"Error validating move: {e}",
                position=from_position
            )


async def main():
    grid = """
    SFFF
    HHFF
    FFFH
    FFFG
    """ 
    steps = 15
    conversation_messages = None
    
    # Initialize game dependencies
    game_deps = GameDeps()

    for i in range(steps):
        if conversation_messages is None:
            result = await move_agent.run(grid, deps=game_deps)
        else:
            result = await move_agent.run(grid, message_history=conversation_messages, deps=game_deps)

        print(f"Step {i+1}: action='{result.output.action.action}'")
        print(f"Reward: {result.output.reward}")
        print(f"Current state: {result.output.current_state}")
        print(f"Reward: {result.output.reward}")
        print(f"Selected actions so far: {result.output.selected_actions}")
        print(f"Path taken so far: {[f'({p.x},{p.y})' for p in result.output.path_taken]}")

        if result.output.terminal_state:
            print(f"Game over at step {i+1}")
            break

        conversation_messages = result.all_messages()
    
    # Pretty print the results
    #pretty_print_grid_data(result.output, grid)
    
    # Also show the raw JSON for debugging
    print("\n🔧 RAW JSON DATA:")
    print("-" * 20)
    print(json.dumps(result.output.model_dump(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
