# Small Agents

A collection of AI agents built with `pydantic_ai` framework, demonstrating various capabilities from weather forecasting to interactive chat.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd small-agents

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pydantic-ai logfire httpx python-dotenv devtools
```

### Environment Setup
Create a `.env` file in the root directory:
```env
# Optional: For weather agent
WEATHER_API_KEY=your_tomorrow_io_api_key
GEO_API_KEY=your_geocode_api_key

# Optional: For logfire monitoring
LOGFIRE_TOKEN=your_logfire_token
```

## 🤖 Agents Overview

### 1. Weather Agent (`src/01_weather_agent.py`)
- **Purpose**: Get real-time weather information for multiple locations
- **Features**: 
  - Geocoding support
  - Multi-location weather data
  - Trip planning with weather insights
- **Usage**: `python src/01_weather_agent.py`

### 2. Research Agent (`src/02_research_agent.py`)
- **Purpose**: Research and information gathering
- **Status**: In development

### 3. Tic Tac Toe Agent (`src/03_tick_tock_agent.py`)
- **Purpose**: Play Tic Tac Toe with AI
- **Features**: Two-player game simulation
- **Usage**: `python src/03_tick_tock_agent.py`

### 4. General Chat Agent (`src/04_general_chat_agent.py`) ⭐
- **Purpose**: Interactive conversational AI assistant
- **Features**:
  - Casual conversations
  - User information memory
  - Mathematical calculations
  - Current time queries
  - Step-by-step problem solving
- **Usage**: 
  ```bash
  # Demo mode
  python src/04_general_chat_agent.py
  
  # Interactive mode
  python src/04_general_chat_agent.py --interactive
  ```

## 🎯 General Chat Agent Deep Dive

### Features
- **Conversation Memory**: Maintains context across the entire session
- **Tool Integration**: Built-in tools for calculations, time, and user info
- **Interactive Mode**: Real-time conversation with command support
- **Logging**: Comprehensive logfire integration for monitoring

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Tools
    participant L as Logfire
    participant G as Gemini Model

    Note over U,G: Interactive Mode Session Start
    
    U->>A: Start interactive mode
    A->>L: Create session span
    A->>U: Display welcome message
    
    loop Conversation Loop
        U->>A: Send user input
        A->>L: Create message span
        A->>L: Log input attributes
        
        alt Has conversation history
            A->>G: Run with message_history
        else First message
            A->>G: Run without history
        end
        
        G->>A: Return response
        
        alt Tool call needed
            A->>T: Execute tool (calculate/time/remember/think)
            T->>L: Create tool span
            T->>L: Log tool attributes
            T->>A: Return tool result
            A->>G: Continue with tool result
            G->>A: Return final response
        end
        
        A->>L: Log response attributes
        A->>U: Display response
        A->>A: Update conversation_messages = result.all_messages()
    end
    
    U->>A: Send quit command
    A->>L: Log session end
    A->>U: Display goodbye message
```

### Tool Execution Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Tool
    participant L as Logfire
    participant G as Gemini Model

    A->>G: Process user input
    G->>A: Decide to use tool
    
    A->>T: Call tool with parameters
    T->>L: Create tool span
    T->>L: Set input attributes
    
    alt Tool execution
        T->>T: Execute tool logic
        T->>L: Set result attributes
        T->>A: Return tool result
    else Tool error
        T->>L: Set error attributes
        T->>A: Return error message
    end
    
    A->>G: Continue with tool result
    G->>A: Generate final response
    A->>L: Log response metrics
```

### Tools Available
1. **`remember_user_info`**: Store user name and interests
2. **`get_current_time`**: Get current date and time
3. **`calculate_simple`**: Perform basic mathematical operations
4. **`get_thinking`**: Step-by-step problem analysis

### Usage Examples

#### Demo Mode
```bash
python src/04_general_chat_agent.py
```
Runs predefined conversations showcasing all agent capabilities.

#### Interactive Mode
```bash
python src/04_general_chat_agent.py --interactive
```

**Available Commands:**
- `help` - Show available commands
- `quit`, `exit`, `bye` - End conversation
- Any other input - Chat with the agent

**Example Conversation:**
```
👤 You: Hi! My name is Alice and I love reading.
🤖 Assistant: Nice to meet you, Alice! I'll remember that you enjoy reading.

👤 You: What do you remember about me?
🤖 Assistant: I remember that your name is Alice and you love reading!

👤 You: Can you calculate 15 * 7 + 23?
🤖 Assistant: The answer is 128.

👤 You: What time is it right now?
🤖 Assistant: The current time is 2025-07-01 14:30:00.
```

## 📊 Monitoring & Logging

All agents use **logfire** for comprehensive monitoring:

### Logfire Features
- **Automatic HTTP request logging** (when AsyncClient is used)
- **Tool execution spans** with detailed attributes
- **Conversation session tracking**
- **Performance metrics** (response times, token usage)
- **Error tracking** and debugging information

### Logfire Configuration
```python
# Configure logfire (sends data only if token is present)
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
```

## 🏗️ Architecture

### Agent Structure
Each agent follows a consistent pattern:
1. **Dependencies**: External APIs, clients, configuration
2. **Tools**: Specialized functions with `@agent.tool` decorator
3. **Agent Configuration**: Model, system prompt, output types
4. **Main Logic**: Async execution with proper error handling

### Key Components
- **`pydantic_ai.Agent`**: Core agent framework
- **`httpx.AsyncClient`**: Asynchronous HTTP client
- **`logfire`**: Observability and monitoring
- **`pydantic.BaseModel`**: Data validation and serialization

## 🔧 Development

### Adding New Tools
```python
@agent.tool
async def my_new_tool(ctx: RunContext, param: str) -> str:
    """Tool description."""
    with logfire.span("my_new_tool") as span:
        # Tool logic here
        span.set_attribute("result", "success")
        return "Tool result"
```

### Adding New Agents
1. Create new file in `src/` directory
2. Follow the established pattern from existing agents
3. Add proper logging and error handling
4. Update this README with documentation

## 📈 Performance

### Optimization Tips
- Use `AsyncClient` for concurrent HTTP requests
- Implement proper error handling with `ModelRetry`
- Use logfire spans for performance monitoring
- Leverage conversation memory for context-aware responses

### Monitoring Metrics
- Response times per tool
- Token usage and costs
- Error rates and types
- User interaction patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your agent or improvements
4. Include proper documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pydantic_ai**: AI agent framework
- **logfire**: Observability platform
- **httpx**: Async HTTP client
- **Gemini 2.0 Flash**: AI model powering the agents 