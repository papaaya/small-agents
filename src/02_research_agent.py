from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from openai.types.responses import WebSearchToolParam  
from pydantic import BaseModel, Field
from devtools import debug
import logfire
import dotenv

dotenv.load_dotenv()

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model=model, 
                model_settings=model_settings,
                system_prompt="""
                You are a helpful assistant that can search the web for information.
                Depending on the complexity of the questions you can choose to adopt 3 differnet strategies to answer the question:
                1. If the question is simple, you can answer it directly.
                2. If the question is complex, you can search the web for information.
                3. If the question is very complex, generate some sub-questions to answer the question.
                    - You can use the web_search_preview tool to search the web for information for sub questions.
                    - sub questions should be as specific as possible.
                    - or sub question can be breakdown of the main question.
                    - or sub question can be fundamental questions to answer the main question.
                    - you must generate at least 5 sub questions.
                4. Deep progressive Research: This involves a more in-depth analysis of the information gathered from the web search.
                    for this strategy you have to progressively research on topic/question collect information, check if you need to ask more question using deep_think tool and answer the question.
                    you can call deep_think tool max 7 times.
            """,
                output_type=str)


@agent.tool
async def deep_think(ctx: RunContext, 
                user_query: str = Field(description="The user's query"), 
                list_of_tools: list[str] = Field(description="The list of tools to use")) -> str:
    """Deep think about the user's query.
    """
    model = OpenAIResponsesModel('gpt-4o')
    settings = OpenAIResponsesModelSettings(
            openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
        )
    agent = Agent(model, model_settings=settings)
    result = await agent.run(user_query)
    return result.output



query = 'What is the weather in Tokyo?'
query_planning = "Generate me a marketing plan for a my new perfume that I am launching in the market. target audience is women between 25-35 years old. and the perfume is for women who want to feel confident"
result = agent.run_sync(query_planning)
print(result.output)



