import asyncio
import argparse
from typing import Any
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings, WebSearchToolParam
from pydantic import BaseModel, Field
from devtools import debug
import logfire
from httpx import AsyncClient

import dotenv

dotenv.load_dotenv()

class ChatResponse(BaseModel):
    message: str = Field(description="The response message")
    confidence: float = Field(description="Confidence level of the response (0-1)")

class UserInfo(BaseModel):
    name: str = Field(description="User's name")
    preferences: list[str] = Field(description="User's preferences or interests")

# Configure logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# Create the general chat agent
agent = Agent(
    "openai:gpt-4o", # optionally google-gla:gemini-2.0-flash
    system_prompt="""
    You are a helpful and friendly general chat assistant. You can:
    - Have casual conversations
    - Answer general questions
    - Remember user information
    - Provide helpful advice
    - Be engaging and conversational
    
    Always be polite, helpful, and try to remember details about the user.
    """,
)

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel("gpt-4o")

web_agent = Agent(model=model, model_settings=model_settings)


@agent.tool
async def remember_user_info(ctx: RunContext, name: str = Field(description="User's name"), 
                           interests: list[str] = Field(description="User's interests")) -> UserInfo:
    """Remember information about the user."""
    with logfire.span("remember_user_info", 
                     attributes={"user.name": name, "user.interests": interests}) as span:
        user_info = UserInfo(
            name=name,
            preferences=interests
        )
        span.set_attribute("user_info_created", True)
        return user_info

@agent.tool
async def get_current_time(ctx: RunContext) -> str:
    """Get the current time."""
    import datetime
    with logfire.span("get_current_time") as span:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        span.set_attribute("current_time", formatted_time)
        span.set_attribute("timezone", str(current_time.tzinfo))
        return f"Current time: {formatted_time}"

@agent.tool
async def calculate_simple(ctx: RunContext, expression: str = Field(description="Simple mathematical expression")) -> str:
    """Calculate simple mathematical expressions."""
    with logfire.span("calculate_simple", attributes={"expression": expression}) as span:
        try:
            # Only allow safe operations
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                span.set_attribute("error", "Invalid characters in expression")
                return "Error: Only basic math operations are allowed"
            
            result = eval(expression)
            span.set_attribute("calculation_result", result)
            span.set_attribute("calculation_success", True)
            return f"Result: {result}"
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("calculation_success", False)
            return f"Error calculating: {str(e)}"

@agent.tool
async def get_thinking(ctx: RunContext, 
                      user_query: str = Field(description="The user's query"), 
                      context: str = Field(description="Current conversation context")) -> str:
    """Allow the agent to think through problems step by step."""
    
    with logfire.span("get_thinking", 
                     attributes={"user_query": user_query, "context_length": len(context)}) as span:
        think_agent = Agent(
            "google-gla:gemini-2.0-flash",
            system_prompt=f"""
            You are a thinking assistant. Analyze the user's query and provide a step-by-step plan.
            Current context: {context}
            
            Think through:
            1. What is the user asking?
            2. What information do we need?
            3. What tools might be helpful?
            4. How should we respond?
            """
        )
        response = await think_agent.run(f"User query: {user_query}")
        span.set_attribute("thinking_response_length", len(response.output))
        return response.output

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='General Chat Agent')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode for continuous conversation')
    args = parser.parse_args()
    
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        
        if args.interactive:
            await run_interactive_mode()
        else:
            await run_demo_mode()

async def run_demo_mode():
    """Run the agent in demo mode with predefined conversations."""
    print("🤖 Chat Agent Started (Demo Mode)!")
    print("=" * 50)
    
    with logfire.span("demo_mode_session") as demo_span:
        # First conversation
        with logfire.span("demo_message_1") as span1:
            result = await agent.run(
                'Hi! My name is John and I love programming and hiking. How are you today?'
            )
            span1.set_attribute("response_length", len(result.output))
            debug(result)
            print('Response:', result.output)
        
        # Continue conversation with context
        with logfire.span("demo_message_2") as span2:
            result = await agent.run(
                'What do you remember about me?'
            )
            span2.set_attribute("response_length", len(result.output))
            debug(result)
            print('Response:', result.output)
        
        # Ask for calculation
        with logfire.span("demo_message_3") as span3:
            result = await agent.run(
                'Can you calculate 15 * 7 + 23?'
            )
            span3.set_attribute("response_length", len(result.output))
            debug(result)
            print('Response:', result.output)
        
        # Ask for current time
        with logfire.span("demo_message_4") as span4:
            result = await agent.run(
                'What time is it right now?'
            )
            span4.set_attribute("response_length", len(result.output))
            debug(result)
            print('Response:', result.output)
        
        demo_span.set_attribute("demo_completed", True)

        # Ask for web search
        with logfire.span("demo_message_5") as span5:
            result = await web_agent.run(
                'What is the weather in Tokyo?'
            )
            span5.set_attribute("response_length", len(result.output))
            debug(result)
            print('Response:', result.output)

async def run_interactive_mode():
    """Run the agent in interactive mode for continuous conversation."""
    print("🤖 Chat Agent Started (Interactive Mode)!")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'help' to see available commands.")
    print("-" * 50)
    
    conversation_messages = None
    message_count = 0
    
    with logfire.span("interactive_conversation_session") as session_span:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    session_span.set_attribute("session_ended", "user_quit")
                    session_span.set_attribute("total_messages", message_count)
                    print("\n🤖 Goodbye! Have a great day!")
                    break
                    
                if user_input.lower() == 'help':
                    print("\n📋 Available Commands:")
                    print("- Type any message to chat with the agent")
                    print("- 'quit', 'exit', or 'bye' to end conversation")
                    print("- 'help' to show this help message")
                    print("- The agent can remember your info, calculate math, and tell time!")
                    continue
                    
                if not user_input:
                    continue

                message_count += 1
                with logfire.span("user_message", 
                                attributes={"message_number": message_count, 
                                          "input_length": len(user_input),
                                          "has_context": conversation_messages is not None}) as msg_span:
                    
                    if conversation_messages is not None:
                        result = await agent.run(user_input, message_history=conversation_messages)
                    else:
                        result = await agent.run(user_input)

                    msg_span.set_attribute("response_length", len(result.output))
                    
                    print(f"\n🤖 Assistant: {result.output}")

                    conversation_messages = result.all_messages()

            except KeyboardInterrupt:
                session_span.set_attribute("session_ended", "keyboard_interrupt")
                session_span.set_attribute("total_messages", message_count)
                print("\n\n🤖 Goodbye! Have a great day!")
                break
            except Exception as e:
                session_span.set_attribute("error", str(e))
                print(f"\n❌ Error: {e}")
                print("Please try again.")

if __name__ == '__main__':
    asyncio.run(main())
