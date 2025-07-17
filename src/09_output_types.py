import asyncio
from typing import Union

from pydantic_ai import Agent

from typing import Union

from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Union[list[str], list[int]],  
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.output)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.output)
#> [10, 20, 30]


from pydantic_ai import Agent, TextOutput


def split_into_words(text: str) -> list[str]:
    return text.split()


agent2 = Agent(
    'openai:gpt-4o',
    output_type=TextOutput(split_into_words),
)
result = agent2.run_sync('Who was Albert Einstein?, in 10 words')
print(result.output)
#> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
