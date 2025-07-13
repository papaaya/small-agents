"""
Prompt Optimization Agent using DSPy and pydantic_ai

This agent uses DSPy for prompt optimization while maintaining
pydantic_ai as the main agent framework.
"""

import dspy
from dspy.teleprompt import SIMBA
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic import BaseModel, Field
import logfire
import dotenv
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import openai

dotenv.load_dotenv()
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# ============================================================================
# SAMPLE DATASET CREATION
# ============================================================================

def create_sample_dataset():
    """Create a sample dataset for prompt optimization testing."""
    
    # Sample task: Text classification (sentiment analysis) - Expanded dataset
    training_data = [
        {
            "input": "I love this movie! It's absolutely fantastic.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'love', 'fantastic' and exclamation marks indicating enthusiasm."
        },
        {
            "input": "This restaurant has terrible service and awful food.",
            "output": "negative", 
            "reasoning": "The text contains negative words like 'terrible', 'awful' indicating dissatisfaction."
        },
        {
            "input": "The weather is okay today, nothing special.",
            "output": "neutral",
            "reasoning": "The text uses neutral language like 'okay', 'nothing special' without strong positive or negative sentiment."
        },
        {
            "input": "Amazing performance! Best concert I've ever been to!",
            "output": "positive",
            "reasoning": "The text contains highly positive words like 'amazing', 'best' and exclamation marks showing excitement."
        },
        {
            "input": "I'm so disappointed with this product. Complete waste of money.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'disappointed', 'waste' indicating frustration and regret."
        },
        {
            "input": "The new coffee shop is wonderful! Great atmosphere and friendly staff.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'wonderful', 'great', 'friendly' indicating satisfaction."
        },
        {
            "input": "This software is completely broken and unusable.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'broken', 'unusable' indicating frustration."
        },
        {
            "input": "The meeting starts at 3 PM in conference room A.",
            "output": "neutral",
            "reasoning": "The text is purely factual without any emotional content."
        },
        {
            "input": "Incredible sunset tonight! The colors are breathtaking.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'incredible', 'breathtaking' indicating awe and appreciation."
        },
        {
            "input": "The customer service was horrible and they refused to help.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'horrible', 'refused' indicating poor service experience."
        },
        {
            "input": "The package arrived on time as expected.",
            "output": "neutral",
            "reasoning": "The text is factual and neutral, just stating what happened."
        },
        {
            "input": "This book is absolutely brilliant! I couldn't put it down.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'brilliant' and enthusiastic language indicating high praise."
        },
        {
            "input": "The food was disgusting and I couldn't eat it.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'disgusting' indicating strong dislike."
        },
        {
            "input": "The train departs from platform 2 at 10:30 AM.",
            "output": "neutral",
            "reasoning": "The text is purely informational without emotional content."
        },
        {
            "input": "Fantastic news! We won the competition!",
            "output": "positive",
            "reasoning": "The text contains positive words like 'fantastic' and exclamation marks showing excitement."
        },
        {
            "input": "The hotel room was filthy and smelled terrible.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'filthy', 'terrible' indicating poor conditions."
        },
        {
            "input": "The document contains 15 pages of technical specifications.",
            "output": "neutral",
            "reasoning": "The text is purely factual describing document content."
        },
        {
            "input": "This is the best day ever! Everything is going perfectly!",
            "output": "positive",
            "reasoning": "The text contains superlative positive words like 'best', 'perfectly' with exclamation marks."
        },
        {
            "input": "The service was appalling and the staff was rude.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'appalling', 'rude' indicating poor service."
        },
        {
            "input": "The library is open from 9 AM to 6 PM daily.",
            "output": "neutral",
            "reasoning": "The text is purely factual about operating hours."
        },
        {
            "input": "What an incredible experience! I'm so grateful for this opportunity.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'incredible', 'grateful' showing appreciation."
        },
        {
            "input": "The product quality is abysmal and not worth the money.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'abysmal' indicating poor quality."
        },
        {
            "input": "The conference will be held in the main auditorium.",
            "output": "neutral",
            "reasoning": "The text is purely factual about event location."
        },
        {
            "input": "This is absolutely perfect! I love everything about it!",
            "output": "positive",
            "reasoning": "The text contains superlative positive words like 'perfect', 'love' with exclamation marks."
        },
        {
            "input": "The experience was dreadful and I want a refund.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'dreadful' and demand for refund indicating dissatisfaction."
        },
        {
            "input": "The building has 25 floors and 3 elevators.",
            "output": "neutral",
            "reasoning": "The text is purely factual about building specifications."
        },
        {
            "input": "Outstanding work! You've exceeded all expectations!",
            "output": "positive",
            "reasoning": "The text contains positive words like 'outstanding', 'exceeded' showing high praise."
        },
        {
            "input": "The food was inedible and made me sick.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'inedible', 'sick' indicating health issues."
        },
        {
            "input": "The museum exhibits artifacts from ancient civilizations.",
            "output": "neutral",
            "reasoning": "The text is purely factual about museum content."
        },
        {
            "input": "This is phenomenal! I'm blown away by the quality!",
            "output": "positive",
            "reasoning": "The text contains positive words like 'phenomenal', 'blown away' showing amazement."
        },
        {
            "input": "The customer support was useless and unhelpful.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'useless', 'unhelpful' indicating poor support."
        },
        {
            "input": "The report contains data from the last fiscal year.",
            "output": "neutral",
            "reasoning": "The text is purely factual about report content."
        }
    ]
    
    development_data = [
        {
            "input": "The new album is pretty good, I enjoyed most of the songs.",
            "output": "positive",
            "reasoning": "The text contains positive words like 'good', 'enjoyed' indicating satisfaction."
        },
        {
            "input": "This book is boring and poorly written.",
            "output": "negative",
            "reasoning": "The text contains negative words like 'boring', 'poorly' indicating dissatisfaction."
        },
        {
            "input": "The meeting was scheduled for 2 PM tomorrow.",
            "output": "neutral",
            "reasoning": "The text is purely factual without any emotional content."
        },
        {
            "input": "The concert was absolutely amazing! Best night ever!",
            "output": "positive",
            "reasoning": "The text contains superlative positive words like 'amazing', 'best' with exclamation marks."
        },
        {
            "input": "The software crashed repeatedly and lost all my data.",
            "output": "negative",
            "reasoning": "The text describes technical problems and data loss indicating frustration."
        },
        {
            "input": "The store is located at 123 Main Street.",
            "output": "neutral",
            "reasoning": "The text is purely factual about location information."
        }
    ]

    # Assert every item has 'reasoning' field
    for i, item in enumerate(training_data):
        assert 'reasoning' in item, f"Missing 'reasoning' in training_data at index {i}: {item}"
    for i, item in enumerate(development_data):
        assert 'reasoning' in item, f"Missing 'reasoning' in development_data at index {i}: {item}"
    
    return training_data, development_data

# ============================================================================
# SIMBA OPTIMIZATION MODULE
# ============================================================================

class SentimentClassifier(dspy.Signature):
    """A sentiment classifier that takes text and returns sentiment with reasoning."""
    
    text = dspy.InputField(desc="The text to classify")
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")
    reasoning = dspy.OutputField(desc="Explanation for the sentiment classification")

def exact_match_metric(example, pred, trace=None):
    """Exact match metric for sentiment classification."""
    return example.sentiment.lower() == pred.sentiment.lower()

class SIMBAOptimizer:
    """SIMBA-based prompt optimizer using DSPy."""
    
    def __init__(self, optimizer_models: List[str], target_models: List[str], agent=None):
        self.optimizer_models = optimizer_models
        self.target_models = target_models
        self.agent = agent
        
    def create_dspy_datasets(self, training_data: List[Dict], dev_data: List[Dict]):
        """Convert our data format to DSPy datasets."""
        
        # Convert to DSPy format
        train_examples = []
        for item in training_data:
            train_examples.append(dspy.Example(
                text=item["input"],
                sentiment=item["output"],
                reasoning=item["reasoning"]
            ).with_inputs("text"))
        
        dev_examples = []
        for item in dev_data:
            dev_examples.append(dspy.Example(
                text=item["input"],
                sentiment=item["output"],
                reasoning=item["reasoning"]
            ).with_inputs("text"))
        
        return train_examples, dev_examples
    
    def evaluate_prompt_with_simba(self, prompt_module, dev_examples, model_name: str):
        """Evaluate a prompt module using SIMBA-compatible evaluation."""
        try:
            # Use the optimized prompt module from SIMBA
            correct = 0
            total = len(dev_examples)
            
            for example in dev_examples:
                try:
                    # Use the optimized prompt module to make predictions
                    prediction = prompt_module(text=example.text)
                    
                    # Extract sentiment from the prediction
                    predicted_sentiment = prediction.sentiment.lower()
                    expected_sentiment = example.sentiment.lower()
                    
                    # Check if prediction matches expected
                    if predicted_sentiment == expected_sentiment:
                        correct += 1
                        
                except Exception as e:
                    print(f"Error evaluating example: {e}")
                    pass
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error evaluating with {model_name}: {e}")
            return 0.0
    
    def optimize_prompt_with_simba(self, training_data: List[Dict], dev_data: List[Dict], max_steps: int = 8):
        """Optimize prompt using DSPy SIMBA algorithm."""
        
        # Convert data to DSPy format
        train_examples, dev_examples = self.create_dspy_datasets(training_data, dev_data)
        
        best_prompt = None
        best_score = 0.0
        
        try:
            # Use OpenAI client directly for optimization
            client = openai.OpenAI()
            
            # Create initial prompt module
            prompt_module = dspy.ChainOfThought(SentimentClassifier)
            
            # Configure DSPy with a proper LLM before running SIMBA
            dspy.configure(lm=dspy.LM(model='gpt-4'))
            # Use SIMBA for optimization with adjusted parameters for smaller dataset
            simba_optimizer = SIMBA(
                metric=exact_match_metric,
                bsize=8,  # Reduced from 32 to work with smaller dataset
                num_candidates=4,  # Reduced from 6
                max_steps=max_steps,
                max_demos=3,  # Reduced from 4
                demo_input_field_maxlen=100000,
                temperature_for_sampling=0.2,
                temperature_for_candidates=0.2
            )
            
            print(f"Starting SIMBA optimization with {len(train_examples)} training examples...")
            
            # Compile the prompt using SIMBA
            optimized_prompt = simba_optimizer.compile(prompt_module, trainset=train_examples)
            
            print("SIMBA optimization completed!")
            
            # Evaluate the optimized prompt
            score = self.evaluate_prompt_with_simba(optimized_prompt, dev_examples, "gpt-3.5-turbo")
            print(f"SIMBA optimization score: {score:.3f}")
            
            best_prompt = optimized_prompt
            best_score = score
                    
        except Exception as e:
            print(f"Error in SIMBA optimization: {e}")
            # Fallback: create a simple prompt
            best_prompt = dspy.ChainOfThought(SentimentClassifier)
            best_score = 0.0
        
        return best_prompt, best_score

# ============================================================================
# PYDANTIC_AI AGENT
# ============================================================================

class OptimizationRequest(BaseModel):
    """Request model for prompt optimization."""
    task_description: str = Field(description="Description of the task to optimize prompts for")
    training_data: List[Dict[str, Any]] = Field(description="Training data for optimization")
    development_data: List[Dict[str, Any]] = Field(description="Development data for evaluation")
    optimizer_models: List[str] = Field(description="List of models to use for optimization", default=["gpt-4"])
    target_models: List[str] = Field(description="List of target models for evaluation", default=["gpt-3.5-turbo"])
    max_steps: int = Field(description="Maximum optimization steps", default=8)

class OptimizationResult(BaseModel):
    """Result model for prompt optimization."""
    best_score: float = Field(description="Best average score achieved")
    optimized_prompt: str = Field(description="The optimized prompt")
    evaluation_details: Dict[str, float] = Field(description="Scores for each target model")
    optimization_log: List[str] = Field(description="Log of optimization process")

# Initialize pydantic_ai agent
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(
    model=model,
    system_prompt="""
    You are a prompt optimization specialist. You help optimize prompts for various NLP tasks
    using advanced optimization techniques. You can analyze training data, suggest improvements,
    and evaluate prompt performance across different language models.
    """,
    output_type=OptimizationResult
)

@agent.tool
async def optimize_prompt_with_simba(
    ctx: RunContext,
    request: OptimizationRequest
) -> OptimizationResult:
    """Optimize prompts using DSPy SIMBA algorithm."""
    
    with logfire.span("optimize_prompt_with_simba") as span:
        span.set_attribute("task_description", request.task_description)
        span.set_attribute("num_training_examples", len(request.training_data))
        span.set_attribute("num_dev_examples", len(request.development_data))
        
        # Pass the agent to the optimizer
        optimizer = SIMBAOptimizer(
            optimizer_models=request.optimizer_models,
            target_models=request.target_models,
            agent=agent
        )
        
        best_prompt, best_score = optimizer.optimize_prompt_with_simba(
            training_data=request.training_data,
            dev_data=request.development_data,
            max_steps=request.max_steps
        )
        
        optimized_prompt_str = str(best_prompt) if best_prompt else "No optimized prompt found"
        evaluation_details = {"gpt-3.5-turbo": best_score}
        
        return OptimizationResult(
            best_score=best_score,
            optimized_prompt=optimized_prompt_str,
            evaluation_details=evaluation_details,
            optimization_log=[f"SIMBA optimization completed with best score: {best_score:.3f}"]
        )

@agent.tool
async def create_sample_optimization_task(
    ctx: RunContext,
    task_type: str = Field(description="Type of task: sentiment, classification, summarization", default="sentiment")
) -> OptimizationRequest:
    """Create a sample optimization task for testing."""
    
    with logfire.span("create_sample_optimization_task") as span:
        span.set_attribute("task_type", task_type)
        
        # Create sample dataset
        training_data, development_data = create_sample_dataset()
        
        return OptimizationRequest(
            task_description=f"Sentiment analysis for {task_type} classification using SIMBA",
            training_data=training_data,
            development_data=development_data,
            optimizer_models=["gpt-4"],
            target_models=["gpt-3.5-turbo"],
            max_steps=8
        )



# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main function to test the prompt optimization agent."""
    
    print("🚀 Prompt Optimization Agent with SIMBA")
    print("=" * 50)
    
    # Use the agent to orchestrate the entire optimization process
    print("Running SIMBA prompt optimization through pydantic_ai agent...")
    
    # Create the optimization request through the agent
    result = await agent.run("""
    Create a sample optimization task for sentiment analysis and then optimize the prompt using the SIMBA algorithm.
    The task should use GPT-4 for optimization and GPT-3.5-turbo for evaluation.
    Use SIMBA with the standard parameters for prompt optimization.
    """)
    
    # Display results
    print("\n🎯 SIMBA Optimization Results:")
    print(f"   - Best Score: {result.output.best_score:.3f}")
    print(f"   - Optimized Prompt: {result.output.optimized_prompt}...")
    print(f"   - Evaluation Details: {result.output.evaluation_details}")
    print(f"   - Optimization Log: {result.output.optimization_log}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())