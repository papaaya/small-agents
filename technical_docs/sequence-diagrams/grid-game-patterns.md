# Grid Game Patterns: Sequence Diagrams

This document provides sequence diagrams for the three main grid game patterns identified in the Small Agents project.

## Pattern 1: Navigation & Pathfinding (Frozen Lake Style)

### Overview
The navigation pattern focuses on moving an agent from a starting position to a goal while avoiding obstacles.

### Sequence Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant FP as FindPosition
    participant VM as ValidateMove
    participant MP as MovePlayer
    participant CS as CheckSuccess

    U->>A: Provide grid state
    A->>FP: find_player_position(grid)
    FP-->>A: player_pos (x, y)
    A->>FP: find_goal_position(grid)
    FP-->>A: goal_pos (x, y)
    
    loop Until goal reached or failure
        A->>A: Plan next move
        A->>VM: validate_move(from_pos, action)
        VM-->>A: MoveValidation(valid, new_pos, reason)
        
        alt Move is valid
            A->>MP: move_player_agent(grid, action)
            MP-->>A: updated_grid
            A->>CS: check_terminal_state(updated_grid)
            CS-->>A: TerminalState(is_terminal, type, message)
            
            alt Goal reached
                A-->>U: Success - Goal reached
            else Still in progress
                A->>A: Continue to next move
            end
        else Move is invalid
            A->>A: Choose alternative action
        end
    end
```

### Key Interactions
1. **Position Discovery**: Agent finds current position and goal
2. **Move Planning**: Agent determines next action
3. **Validation**: System checks if move is legal and safe
4. **Execution**: Move is applied and grid updated
5. **Success Check**: System determines if goal reached

## Pattern 2: Puzzle Solving & State Transformation (Sokoban Style)

### Overview
The puzzle solving pattern focuses on transforming grid states according to game rules to achieve a solution.

### Sequence Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant AS as AnalyzeState
    participant VA as ValidateAction
    participant UG as UpdateGrid
    participant GR as GetReward
    participant VS as VerifySolution

    U->>A: Provide puzzle state
    A->>AS: analyze_grid_state(grid)
    AS-->>A: GridAnalysis(player_pos, boxes, targets, etc.)
    
    loop Until puzzle solved or stuck
        A->>A: Generate action hypothesis
        A->>VA: valid_action(action)
        VA-->>A: ActionValidation(valid, reason)
        
        alt Action is valid
            A->>UG: update_grid(grid, action, positions)
            UG-->>A: updated_grid, success_message
            A->>GR: get_reward(state, action)
            GR-->>A: reward_value, reasoning
            A->>VS: verify_solution(updated_grid)
            VS-->>A: SolutionStatus(solved, progress)
            
            alt Puzzle solved
                A-->>U: Success - Puzzle solved
            else Still solving
                A->>A: Continue with next action
            end
        else Action is invalid
            A->>A: Generate alternative action
        end
        
        alt Stuck or too many iterations
            A->>A: restart_puzzle(original_state)
            A->>A: Reset to original state
        end
    end
```

### Key Interactions
1. **State Analysis**: Agent understands current puzzle state
2. **Action Generation**: Agent proposes valid actions
3. **State Transformation**: Actions are applied to transform grid
4. **Reward Calculation**: System evaluates action quality
5. **Solution Verification**: Check if puzzle is solved
6. **Restart Mechanism**: Reset if stuck

## Pattern 3: Visual Reasoning & Pattern Recognition (ARC Style)

### Overview
The visual reasoning pattern focuses on identifying and applying transformations to solve visual puzzles.

### Sequence Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant GA as GridAnalysis
    participant PH as PatternHypothesis
    participant AT as ApplyTransformation
    participant CV as CompareValidation
    participant CS as ConfidenceScoring

    U->>A: Provide input grid + examples
    A->>GA: analyze_grid(input_grid, "input")
    GA-->>A: GridAnalysis(shape, values, patterns)
    A->>GA: analyze_grid(output_grid, "output")
    GA-->>A: GridAnalysis(shape, values, patterns)
    
    loop Until solution found or confidence low
        A->>PH: generate_hypothesis(input_analysis, output_analysis)
        PH-->>A: TransformationHypothesis(operations, confidence)
        
        A->>AT: apply_transformation(input_grid, operation)
        AT-->>A: transformed_grid
        
        A->>CV: compare_grids(transformed_grid, expected_output)
        CV-->>A: ValidationResult(match, similarity_score)
        
        A->>CS: calculate_confidence(solution, validation_results)
        CS-->>A: confidence_score
        
        alt High confidence match
            A-->>U: Success - Solution found
        else Low confidence
            A->>A: Refine hypothesis
        end
    end
```

### Key Interactions
1. **Pattern Analysis**: Agent analyzes input and output grids
2. **Hypothesis Generation**: Agent proposes transformation operations
3. **Transformation Application**: Operations are applied to input grid
4. **Validation**: Result is compared with expected output
5. **Confidence Scoring**: System assesses solution reliability

## Common Patterns Across All Three

### Error Handling
```mermaid
sequenceDiagram
    participant A as Agent
    participant T as Tool
    participant E as ErrorHandler

    A->>T: Execute tool
    alt Tool succeeds
        T-->>A: Success result
    else Tool fails
        T->>E: Handle error
        E-->>A: Error message + fallback
        A->>A: Retry or alternative approach
    end
```

### State Management
```mermaid
sequenceDiagram
    participant A as Agent
    participant S as StateManager
    participant G as Grid

    A->>S: Get current state
    S->>G: Retrieve grid
    G-->>S: Current grid state
    S-->>A: State with metadata
    
    A->>S: Update state
    S->>G: Apply changes
    G-->>S: Updated grid
    S-->>A: Confirmation
```

### Observability
```mermaid
sequenceDiagram
    participant A as Agent
    participant L as Logger
    participant M as Metrics

    A->>L: Log action start
    L->>M: Record metrics
    
    A->>A: Execute action
    
    A->>L: Log action result
    L->>M: Update metrics
    
    A->>L: Log confidence score
    L->>M: Record confidence
```

## Pattern Selection Guide

### When to Use Navigation Pattern
- ✅ Agent needs to move from point A to point B
- ✅ Obstacles need to be avoided
- ✅ Goal is clearly defined
- ✅ State changes are simple (position updates)

### When to Use Puzzle Solving Pattern
- ✅ Complex game rules govern state changes
- ✅ Multiple objects interact (boxes, targets, etc.)
- ✅ Progress can be measured incrementally
- ✅ Restart capability is needed

### When to Use Visual Reasoning Pattern
- ✅ Input/output examples are provided
- ✅ Geometric transformations are involved
- ✅ Pattern recognition is required
- ✅ Confidence scoring is important

## Implementation Considerations

### Performance
- **Navigation**: Optimize for pathfinding algorithms
- **Puzzle**: Focus on state space exploration
- **Visual**: Prioritize pattern matching efficiency

### Memory Management
- **Navigation**: Minimal state tracking
- **Puzzle**: Deep copy grids for state transformations
- **Visual**: Cache pattern analysis results

### Error Recovery
- **Navigation**: Retry with alternative paths
- **Puzzle**: Restart from original state
- **Visual**: Refine hypotheses iteratively 