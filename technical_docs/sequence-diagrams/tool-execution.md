# Tool Execution Patterns

## Overview
This document describes the standard patterns for tool execution in the Small Agents framework. All tools follow a consistent pattern for observability, error handling, and integration with the agent system.

## Standard Tool Execution Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Tool
    participant L as Logfire
    participant M as Gemini Model
    participant E as External Service

    A->>M: Process user input
    M->>A: Decide to use tool
    
    A->>T: Call tool with parameters
    T->>L: Create tool span
    T->>L: Set input attributes (params, context)
    
    alt Tool requires external service
        T->>E: Make external request
        E->>T: Return response
        
        alt External service success
            T->>T: Process response
            T->>L: Set success attributes
        else External service error
            T->>L: Set error attributes
            T->>T: Handle error gracefully
        end
    else Tool is pure computation
        T->>T: Execute computation
        T->>L: Set computation attributes
    end
    
    T->>L: Set result attributes (output, duration)
    T->>A: Return tool result
    
    A->>M: Continue with tool result
    M->>A: Generate final response
    A->>L: Log response metrics
```

## Tool Definition Pattern

```mermaid
sequenceDiagram
    participant D as Developer
    participant T as Tool
    participant L as Logfire
    participant V as Validation

    D->>T: Define tool function
    T->>T: Add @agent.tool decorator
    
    D->>T: Add logfire span
    T->>L: Create span with tool name
    
    D->>T: Add input validation
    T->>V: Validate parameters
    V->>T: Return validation result
    
    alt Validation passes
        T->>T: Execute tool logic
        T->>L: Set success attributes
    else Validation fails
        T->>L: Set error attributes
        T->>T: Return error response
    end
    
    T->>L: Set output attributes
    T->>D: Return tool result
```

## Error Handling Pattern

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Tool
    participant L as Logfire
    participant E as Error Handler

    A->>T: Call tool
    T->>L: Create tool span
    
    alt Tool execution
        T->>T: Execute tool logic
        T->>L: Set success attributes
        T->>A: Return result
    else Tool error
        T->>L: Set error attributes (error_type, message)
        T->>E: Handle specific error type
        
        alt Recoverable error
            E->>T: Retry or fallback
            T->>L: Log recovery attempt
            T->>A: Return fallback result
        else Non-recoverable error
            E->>T: Return error message
            T->>A: Return error response
        end
    end
    
    A->>L: Log final tool outcome
```

## Tool Composition Pattern

```mermaid
sequenceDiagram
    participant A as Agent
    participant T1 as Tool 1
    participant T2 as Tool 2
    participant T3 as Tool 3
    participant L as Logfire

    A->>T1: Call first tool
    T1->>L: Create span for tool 1
    T1->>T1: Execute logic
    T1->>A: Return result 1
    
    A->>T2: Call second tool with result 1
    T2->>L: Create span for tool 2
    T2->>T2: Execute logic
    T2->>A: Return result 2
    
    A->>T3: Call third tool with result 2
    T3->>L: Create span for tool 3
    T3->>T3: Execute logic
    T3->>A: Return final result
    
    A->>L: Log composition completion
```

## Async Tool Pattern

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Async Tool
    participant L as Logfire
    participant E as External API

    A->>T: Call async tool
    T->>L: Create async span
    
    T->>E: Make async request
    E->>T: Return async response
    
    T->>T: Process async result
    T->>L: Set async attributes
    
    T->>A: Return async result
    A->>L: Log async completion
```

## Tool Validation Pattern

```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Tool
    participant V as Validator
    participant L as Logfire

    A->>T: Call tool with parameters
    T->>V: Validate input parameters
    V->>L: Create validation span
    
    alt Parameters valid
        V->>T: Return validation success
        T->>T: Execute tool logic
        T->>L: Set success attributes
    else Parameters invalid
        V->>T: Return validation error
        T->>L: Set validation error
        T->>A: Return validation error
    end
    
    T->>V: Validate output
    V->>T: Return output validation
    
    T->>A: Return validated result
```

## Key Components

### 1. Tool Decorator
```python
@agent.tool
async def my_tool(ctx: RunContext, param: str) -> str:
    """Tool description for the agent."""
    with logfire.span("my_tool") as span:
        # Tool implementation
        span.set_attribute("param", param)
        result = process(param)
        span.set_attribute("result", result)
        return result
```

### 2. Logfire Integration
- **Automatic Spans**: Every tool call creates a span
- **Input Logging**: Parameters and context are logged
- **Output Logging**: Results and execution time are logged
- **Error Tracking**: Failures and error types are captured

### 3. Error Handling
- **Graceful Degradation**: Tools handle errors gracefully
- **Error Classification**: Distinguish between recoverable and fatal errors
- **Fallback Mechanisms**: Provide alternative results when possible
- **Error Reporting**: Clear error messages for debugging

### 4. Validation
- **Input Validation**: Check parameters before execution
- **Output Validation**: Verify results meet expectations
- **Type Safety**: Use Pydantic models for structured data
- **Constraint Checking**: Validate business rules and limits

## Common Tool Patterns

### 1. Data Fetching Tools
```python
@agent.tool
async def fetch_data(ctx: RunContext, url: str) -> Dict[str, Any]:
    """Fetch data from external API."""
    with logfire.span("fetch_data") as span:
        span.set_attribute("url", url)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                span.set_attribute("status_code", response.status_code)
                return data
        except Exception as e:
            span.set_attribute("error", str(e))
            raise
```

### 2. Computation Tools
```python
@agent.tool
async def calculate(ctx: RunContext, expression: str) -> float:
    """Evaluate mathematical expression."""
    with logfire.span("calculate") as span:
        span.set_attribute("expression", expression)
        try:
            result = eval(expression)  # In production, use safer eval
            span.set_attribute("result", result)
            return result
        except Exception as e:
            span.set_attribute("error", str(e))
            raise
```

### 3. State Management Tools
```python
@agent.tool
async def remember(ctx: RunContext, key: str, value: str) -> str:
    """Store information in memory."""
    with logfire.span("remember") as span:
        span.set_attribute("key", key)
        memory[key] = value
        span.set_attribute("memory_size", len(memory))
        return f"Remembered {key}: {value}"
```

## Performance Considerations

### 1. Tool Latency
- **External Calls**: Use async/await for I/O operations
- **Caching**: Cache frequently accessed data
- **Connection Pooling**: Reuse HTTP connections
- **Timeout Handling**: Set appropriate timeouts

### 2. Resource Management
- **Memory Usage**: Monitor memory consumption
- **Connection Limits**: Limit concurrent external calls
- **Rate Limiting**: Respect API rate limits
- **Cleanup**: Properly close resources

### 3. Monitoring
- **Execution Time**: Track tool performance
- **Success Rates**: Monitor tool reliability
- **Error Patterns**: Identify common failure modes
- **Resource Usage**: Track memory and CPU usage

## Best Practices

1. **Always use spans**: Every tool should create a logfire span
2. **Validate inputs**: Check parameters before processing
3. **Handle errors gracefully**: Provide meaningful error messages
4. **Log key metrics**: Track performance and success rates
5. **Use async when possible**: Improve responsiveness
6. **Document tools clearly**: Help the agent choose the right tool
7. **Test thoroughly**: Ensure tools work reliably
8. **Monitor performance**: Track and optimize slow tools 