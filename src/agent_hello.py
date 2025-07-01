from pydantic_ai import Agent
import dotenv

dotenv.load_dotenv()


agent = Agent("google-gla:gemini-2.0-flash", system_prompt="You are a helpful assistant that greets the user.")

response = agent.run_sync("Hello, how are you?")
print(response)