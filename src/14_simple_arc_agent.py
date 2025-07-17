import asyncio
import json
import numpy as np
from pydantic_ai import Agent, Tool, RunContext
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import colors
import logfire
import dotenv

dotenv.load_dotenv()

# Configure Logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

def log_task_metrics(task_name: str, confidence: float, steps: int, operations: list):
    """Log metrics for a completed task"""
    logfire.info("Task completed", 
                task_name=task_name,
                confidence=confidence,
                num_steps=steps,
                operations_used=operations,
                success=confidence > 0.8)

def log_error(error: Exception, context: str = ""):
    """Log errors with context"""
    logfire.error("ARC agent error", 
                error=str(error),
                error_type=type(error).__name__,
                context=context)
import logfire
import dotenv

dotenv.load_dotenv()

# Configure logfire
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()



# Define the 10 official ARC colors
ARC_COLORMAP = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])

class GridAnalysis(BaseModel):
    """Analysis of a grid pattern"""
    grid_shape: List[int] = Field(description="Shape of the grid (rows, cols)")
    unique_values: List[int] = Field(description="Unique values/colors in the grid")
    value_counts: Dict[str, int] = Field(description="Count of each value")
    pattern_description: str = Field(description="Description of the pattern")

class TransformationStep(BaseModel):
    """A single transformation step"""
    operation: str = Field(description="Name of the operation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation")
    description: str = Field(description="What this step does")

class ARCSolution(BaseModel):
    """Solution for an ARC task"""
    task_name: str = Field(description="Name of the task")
    transformation_steps: List[TransformationStep] = Field(description="Steps to transform input to output")
    final_output: List[List[int]] = Field(description="Predicted output grid")
    confidence: float = Field(description="Confidence in the solution (0-1)")
    reasoning: str = Field(description="Explanation of the solution")

# Simple DSL operations
def rotate_90(grid):
    """Rotate grid 90 degrees clockwise"""
    return np.rot90(grid, -1).tolist()

def flip_horizontal(grid):
    """Flip grid horizontally"""
    return np.fliplr(grid).tolist()

def flip_vertical(grid):
    """Flip grid vertically"""
    return np.flipud(grid).tolist()

def replace_values(grid, old_value, new_value):
    """Replace all instances of old_value with new_value"""
    grid_array = np.array(grid)
    grid_array[grid_array == old_value] = new_value
    return grid_array.tolist()

def add_border(grid, border_value=0):
    """Add a border around the grid"""
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    bordered = np.full((rows + 2, cols + 2), border_value, dtype=int)
    bordered[1:-1, 1:-1] = grid_array
    return bordered.tolist()

def tile_grid(grid, factor=3):
    """Tile/repeat the grid to create a larger grid"""
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    tiled = np.tile(grid_array, (factor, factor))
    return tiled.tolist()

# DSL dictionary
DSL = {
    'rotate_90': rotate_90,
    'flip_h': flip_horizontal,
    'flip_v': flip_vertical,
    'replace': replace_values,
    'add_border': add_border,
    'tile': tile_grid,
}

# Create the ARC solving agent
arc_agent = Agent(
    "openai:gpt-4o",
    output_type=ARCSolution,
    system_prompt="""
    You are an expert at solving ARC (Abstraction and Reasoning Corpus) tasks.
    Your job is to analyze input-output pairs and figure out the transformation rule.
    
    # Available Operations:
    - rotate_90: Rotate the grid 90 degrees clockwise
    - flip_h: Flip the grid horizontally (mirror)
    - flip_v: Flip the grid vertically (mirror)
    - replace: Replace all instances of one value with another
    - add_border: Add a border around the grid
    - tile: Tile/repeat the grid to create a larger grid (factor=3 by default)
    
    # Process:
    1. Analyze the input and output grids carefully
    2. Look for patterns, transformations, and rules
    3. Break down the solution into simple steps
    4. Test your solution on the training examples
    5. Provide clear reasoning for your approach
    
    # Tips:
    - Start with simple operations and build up
    - Look for geometric transformations first
    - Then look for value/color changes
    - Consider the order of operations
    - Verify your solution works on all training examples
    - If output is larger than input, consider tiling operations
    """
)

@arc_agent.tool
async def analyze_grid(ctx: RunContext, grid: List[List[int]], name: str = "grid") -> GridAnalysis:
    """Analyze a grid to understand its properties"""
    with logfire.span("analyze_grid", attributes={"grid_name": name, "grid_shape": list(np.array(grid).shape)}) as span:
        grid_array = np.array(grid)
        
        analysis = GridAnalysis(
            grid_shape=list(grid_array.shape),
            unique_values=sorted(list(set(grid_array.flatten()))),
            value_counts={str(val): int(np.sum(grid_array == val)) for val in set(grid_array.flatten())},
            pattern_description=f"Grid of shape {grid_array.shape} with values {sorted(list(set(grid_array.flatten())))}"
        )
        
        # Log analysis results
        span.set_attribute("grid_shape", analysis.grid_shape)
        span.set_attribute("unique_values", analysis.unique_values)
        span.set_attribute("value_counts", analysis.value_counts)
        span.set_attribute("pattern_description", analysis.pattern_description)
        
        print(f"📊 {name} Analysis:")
        print(f"   Shape: {analysis.grid_shape}")
        print(f"   Values: {analysis.unique_values}")
        print(f"   Counts: {analysis.value_counts}")
        print(f"   Description: {analysis.pattern_description}")
        
        return analysis

@arc_agent.tool
async def apply_transformation(ctx: RunContext, grid: List[List[int]], operation: str, **params) -> List[List[int]]:
    """Apply a transformation to a grid"""
    with logfire.span("apply_transformation", attributes={"operation": operation, "params": params, "input_shape": list(np.array(grid).shape)}) as span:
        if operation not in DSL:
            span.set_attribute("error", f"Unknown operation: {operation}")
            raise ValueError(f"Unknown operation: {operation}")
        
        print(f"🔄 Applying {operation} with params {params}")
        result = DSL[operation](grid, **params) if params else DSL[operation](grid)
        
        output_shape = list(np.array(result).shape)
        span.set_attribute("output_shape", output_shape)
        span.set_attribute("success", True)
        
        print(f"   Input shape: {np.array(grid).shape}")
        print(f"   Output shape: {output_shape}")
        
        return result

@arc_agent.tool
async def compare_grids(ctx: RunContext, grid1: List[List[int]], grid2: List[List[int]], name1: str = "grid1", name2: str = "grid2") -> bool:
    """Compare two grids and return True if they match"""
    with logfire.span("compare_grids", attributes={"grid1_name": name1, "grid2_name": name2, "grid1_shape": list(np.array(grid1).shape), "grid2_shape": list(np.array(grid2).shape)}) as span:
        match = np.array_equal(np.array(grid1), np.array(grid2))
        
        span.set_attribute("match", match)
        span.set_attribute("grid1_shape", list(np.array(grid1).shape))
        span.set_attribute("grid2_shape", list(np.array(grid2).shape))
        
        print(f"🔍 Comparing {name1} and {name2}: {'✅ MATCH' if match else '❌ DIFFERENT'}")
        if not match:
            print(f"   {name1} shape: {np.array(grid1).shape}")
            print(f"   {name2} shape: {np.array(grid2).shape}")
        
        return match

def create_simple_arc_task():
    """Create a simple ARC task for testing"""
    task = {
        "name": "simple_rotation",
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]
                ],
                "output": [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]
                ]
            },
            {
                "input": [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]
                ],
                "output": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            }
        ]
    }
    return task

def load_real_arc_task():
    """Load a real ARC task from the dataset"""
    try:
        task_file = '/Users/hpathak/dev/ARC-AGI/data/training/007bbfb7.json'
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Convert to our format
        task = {
            "name": "007bbfb7",
            "train": task_data.get('train', []),
            "test": task_data.get('test', [])
        }
        return task
    except FileNotFoundError:
        print("⚠️  Real ARC task file not found, using simple task")
        return create_simple_arc_task()
    except Exception as e:
        print(f"⚠️  Error loading real ARC task: {e}")
        return create_simple_arc_task()

async def test_with_real_task():
    """Test the agent with a real ARC task"""
    with logfire.span("test_with_real_task") as span:
        print("🎯 Testing with Real ARC Task")
        print("=" * 50)
        
        # Load the real task
        task = load_real_arc_task()
        span.set_attribute("task_name", task['name'])
        span.set_attribute("num_training_examples", len(task['train']))
        print(f"📋 Task: {task['name']}")
        
        # Show the training examples
        for i, example in enumerate(task['train']):
            print(f"\n📝 Training Example {i+1}:")
            print(f"Input:")
            for row in example['input']:
                print(f"  {row}")
            print(f"Output:")
            for row in example['output']:
                print(f"  {row}")
        
        # Let the agent solve the task
        print(f"\n🤖 Agent solving real ARC task...")
        with logfire.span("agent_solve_real_task") as solve_span:
            try:
                result = await arc_agent.run(f"""
                Solve this real ARC task:
                
                Task: {task['name']}
                
                Training examples:
                {json.dumps(task['train'], indent=2)}
                
                This is a real ARC task from the dataset. Analyze the patterns carefully and provide a step-by-step solution.
                """)
                
                solve_span.set_attribute("confidence", result.output.confidence)
                solve_span.set_attribute("num_steps", len(result.output.transformation_steps))
                solve_span.set_attribute("operations_used", [step.operation for step in result.output.transformation_steps])
            except Exception as e:
                log_error(e, f"Real task: {task['name']}")
                solve_span.set_attribute("error", str(e))
                raise
        
        print(f"\n🎉 Solution Found!")
        print(f"Confidence: {result.output.confidence}")
        print(f"Reasoning: {result.output.reasoning}")
        print(f"Steps:")
        for i, step in enumerate(result.output.transformation_steps):
            print(f"  {i+1}. {step.operation}: {step.description}")
        
        print(f"\nPredicted Output:")
        for row in result.output.final_output:
            print(f"  {row}")
        
        span.set_attribute("final_confidence", result.output.confidence)
        span.set_attribute("solution_steps", len(result.output.transformation_steps))
        
        # Log task metrics
        log_task_metrics(
            task_name=task['name'],
            confidence=result.output.confidence,
            steps=len(result.output.transformation_steps),
            operations=[step.operation for step in result.output.transformation_steps]
        )
        
        return result

async def main():
    """Main function to test the ARC agent"""
    with logfire.span("arc_agent_main") as main_span:
        print("🎯 Simple ARC Agent - Step by Step Learning")
        print("=" * 50)
        
        # Test 1: Simple rotation task
        print("\n🔬 Test 1: Simple Rotation Task...")
        with logfire.span("test_simple_task") as simple_span:
            task = create_simple_arc_task()
            simple_span.set_attribute("task_name", task['name'])
            simple_span.set_attribute("num_examples", len(task['train']))
            print(f"📋 Task: {task['name']}")
            
            # Show the training examples
            for i, example in enumerate(task['train']):
                print(f"\n📝 Training Example {i+1}:")
                print(f"Input:")
                for row in example['input']:
                    print(f"  {row}")
                print(f"Output:")
                for row in example['output']:
                    print(f"  {row}")
            
            # Let the agent solve the task
            print(f"\n🤖 Agent solving simple task...")
            with logfire.span("agent_solve_simple_task") as solve_span:
                try:
                    result1 = await arc_agent.run(f"""
                    Solve this ARC task:
                    
                    Task: {task['name']}
                    
                    Training examples:
                    {json.dumps(task['train'], indent=2)}
                    
                    Analyze the patterns and provide a step-by-step solution.
                    """)
                    
                    solve_span.set_attribute("confidence", result1.output.confidence)
                    solve_span.set_attribute("num_steps", len(result1.output.transformation_steps))
                    solve_span.set_attribute("operations_used", [step.operation for step in result1.output.transformation_steps])
                except Exception as e:
                    log_error(e, f"Simple task: {task['name']}")
                    solve_span.set_attribute("error", str(e))
                    raise
            
            print(f"\n🎉 Solution Found!")
            print(f"Confidence: {result1.output.confidence}")
            print(f"Reasoning: {result1.output.reasoning}")
            print(f"Steps:")
            for i, step in enumerate(result1.output.transformation_steps):
                print(f"  {i+1}. {step.operation}: {step.description}")
            
            print(f"\nPredicted Output:")
            for row in result1.output.final_output:
                print(f"  {row}")
            
            simple_span.set_attribute("final_confidence", result1.output.confidence)
            simple_span.set_attribute("solution_steps", len(result1.output.transformation_steps))
            
            # Log task metrics
            log_task_metrics(
                task_name=task['name'],
                confidence=result1.output.confidence,
                steps=len(result1.output.transformation_steps),
                operations=[step.operation for step in result1.output.transformation_steps]
            )
        
        # Test 2: Real ARC task
        print(f"\n" + "="*60)
        print("\n🔬 Test 2: Real ARC Task...")
        result2 = await test_with_real_task()
        
        # Summary comparison
        print(f"\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Test 1 (Simple): Confidence = {result1.output.confidence:.2f}")
        print(f"Test 2 (Real):   Confidence = {result2.output.confidence:.2f}")
        print(f"\nSimple task steps: {len(result1.output.transformation_steps)}")
        print(f"Real task steps:   {len(result2.output.transformation_steps)}")
        
        # Log summary metrics
        main_span.set_attribute("test1_confidence", result1.output.confidence)
        main_span.set_attribute("test2_confidence", result2.output.confidence)
        main_span.set_attribute("test1_steps", len(result1.output.transformation_steps))
        main_span.set_attribute("test2_steps", len(result2.output.transformation_steps))
        main_span.set_attribute("overall_success", result1.output.confidence > 0.8 and result2.output.confidence > 0.8)
        
        return result1, result2

if __name__ == "__main__":
    asyncio.run(main()) 