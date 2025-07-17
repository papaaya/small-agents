# General Chat Agent - Interactive Conversation Flow

## Overview
The General Chat Agent provides an interactive conversational experience with memory, tools, and comprehensive observability. This document shows the detailed flow of how the agent processes user input and maintains conversation context.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Tools
    participant L as Logfire
    participant G as Gemini Model
    participant M as Memory

    Note over U,M: Session Initialization
    
    U->>A: Start interactive mode
    A->>L: Create session span
    A->>M: Initialize conversation_messages = []
    A->>U: Display welcome message
    
    Note over U,M: Conversation Loop
    
    loop Each User Input
        U->>A: Send user input
        A->>L: Create message span
        A->>L: Log input attributes (length, timestamp)
        
        alt Has conversation history
            A->>M: Retrieve conversation_messages
            A->>G: Run with message_history
        else First message
            A->>G: Run without history
        end
        
        G->>A: Return initial response
        
        alt Tool call needed
            A->>T: Execute tool (calculate/time/remember/think)
            T->>L: Create tool span
            T->>L: Log tool attributes (name, params)
            
            alt Tool execution success
                T->>T: Execute tool logic
                T->>L: Set result attributes
                T->>A: Return tool result
            else Tool execution error
                T->>L: Set error attributes
                T->>A: Return error message
            end
            
            A->>G: Continue with tool result
            G->>A: Return final response
        end
        
        A->>L: Log response attributes (length, confidence)
        A->>U: Display response
        A->>M: Update conversation_messages = result.all_messages()
        A->>L: Log conversation state (message_count)
    end
    
    Note over U,M: Session Termination
    
    U->>A: Send quit command
    A->>L: Log session end
    A->>L: Log final metrics (total_messages, session_duration)
    A->>U: Display goodbye message
```

## Tool Execution Flow

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
    T->>L: Set input attributes (params, context)
    
    alt Tool execution
        T->>T: Execute tool logic
        T->>L: Set result attributes (success, output)
        T->>A: Return tool result
    else Tool error
        T->>L: Set error attributes (error_type, message)
        T->>A: Return error message
    end
    
    A->>G: Continue with tool result
    G->>A: Generate final response
    A->>L: Log response metrics (tokens, confidence)
```

## Memory Management Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant M as Memory
    participant L as Logfire

    U->>A: "My name is Alice"
    A->>M: Store user_info = {"name": "Alice"}
    A->>L: Log memory update
    
    U->>A: "What's my name?"
    A->>M: Retrieve user_info
    A->>A: Generate response using memory
    A->>U: "Your name is Alice!"
    
    U->>A: "I love reading"
    A->>M: Update user_info = {"name": "Alice", "interests": ["reading"]}
    A->>L: Log memory update
```

## Key Components

### 1. Conversation Memory
- **Purpose**: Maintains context across the entire session
- **Implementation**: `conversation_messages` list containing all messages
- **Persistence**: In-memory during session, cleared on restart

### 2. User Memory
- **Purpose**: Stores user preferences and information
- **Implementation**: `user_info` dictionary
- **Tools**: `remember_user_info()` for storage, implicit retrieval

### 3. Tool Integration
- **Available Tools**:
  - `remember_user_info()`: Store user preferences
  - `get_current_time()`: Get current date/time
  - `calculate_simple()`: Basic mathematical operations
  - `get_thinking()`: Step-by-step problem analysis

### 4. Observability
- **Session Tracking**: Complete conversation flow
- **Tool Monitoring**: Execution times and success rates
- **Performance Metrics**: Response times, token usage
- **Error Tracking**: Tool failures and debugging info

## Error Handling

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Tool
    participant L as Logfire

    U->>A: "Calculate 1/0"
    A->>T: calculate_simple("1/0")
    T->>L: Log error (DivisionByZero)
    T->>A: Return error message
    A->>G: Continue with error context
    G->>A: Generate helpful error response
    A->>U: "I can't divide by zero. Try a different calculation."
```

## Performance Considerations

1. **Memory Management**: Conversation history grows with session length
2. **Tool Latency**: External tools may add response time
3. **Token Usage**: Long conversations consume more tokens
4. **Error Recovery**: Graceful handling of tool failures

## Monitoring Metrics

- **Session Duration**: Total time from start to quit
- **Message Count**: Number of user inputs processed
- **Tool Usage**: Frequency and success rate of each tool
- **Response Times**: Average time to generate responses
- **Error Rates**: Percentage of failed tool executions 