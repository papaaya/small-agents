from tarfile import data_filter
from pydantic_ai import Agent, Tool, RunContext
from pydantic import BaseModel, Field
from dataclasses import dataclass
import asyncio
from typing import Any

tic_toc_toe_agent = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="""
    You are an expert tic-tac-toe player.
    You are given a tic-tac-toe board and you need to play the game.
    You need to play the game optimally and win the game.
    """
)

tic_toc_toe_agent_player2 = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="""
    You are an expert tic-tac-toe player.
    You are given a tic-tac-toe board and you need to play the game.
    You need to play the game optimally and win the game.
    """
)

@dataclass
class TicTacToeBoard:
    board: list[list[str]]


@tic_toc_toe_agent.tool
def make_x_move(ctx: RunContext, position_i: int, position_j: int) -> dict[str, Any]:
    """ Add an X to the board to optimially win the game 
        position_i: The row index of the board
        position_j: The column index of the board
        board[position_i][position_j] = "X" should be updated to "X"
    """
    ctx.deps.board[position_i][position_j] = "X"
    
    return {
        "position_i": position_i,
        "position_j": position_j,
        "board": ctx.deps.board
    }


@tic_toc_toe_agent_player2.tool
def make_o_move(ctx: RunContext, position_i: int, position_j: int, winner: str) -> dict[str, Any]:
    """ Add an O to the board to optimially win the game 
        position_i: The row index of the board
        position_j: The column index of the board
        board[position_i][position_j] = "O" should be updated to "O"
        winner: The winner of the game if the game is won by this move. If the game is not won by this move, then winner should be "No winner"
    """
    ctx.deps.board[position_i][position_j] = "O"

    return {
        "position_i": position_i,
        "position_j": position_j,
        "board": ctx.deps.board,
        "winner": winner
    }


async def main():
    
    result1 = await tick_toc_toe_agent.run("What is the best move?", 
        deps=TicTacToeBoard(board=[["", "", ""], ["", "", ""], ["", "", ""]])
        )
    print(result1.output)

    result2 = await tic_toc_toe_agent_player2.run("What is the best move?", messages=result1.new_messages())
    print(result2.output)


if __name__ == "__main__":
    asyncio.run(main())