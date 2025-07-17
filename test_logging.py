#!/usr/bin/env python3
"""
Test script to verify Logfire logging in the ARC agent
"""

import asyncio
import os
import sys
sys.path.append('src')
from 14_simple_arc_agent import main, log_task_metrics, log_error

async def test_logging():
    """Test the logging functionality"""
    print("🧪 Testing Logfire Logging in ARC Agent")
    print("=" * 50)
    
    # Test basic logging functions
    print("\n📝 Testing basic logging functions...")
    
    # Test task metrics logging
    log_task_metrics(
        task_name="test_task",
        confidence=0.95,
        steps=3,
        operations=["rotate_90", "flip_horizontal", "replace_values"]
    )
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        log_error(e, "Test context")
    
    print("✅ Basic logging functions tested")
    
    # Test full agent with logging
    print("\n🤖 Testing full agent with logging...")
    try:
        result1, result2 = await main()
        print("✅ Full agent test completed successfully")
        return result1, result2
    except Exception as e:
        print(f"❌ Full agent test failed: {e}")
        log_error(e, "Full agent test")
        raise

if __name__ == "__main__":
    # Check if LOGFIRE_TOKEN is set
    if not os.getenv("LOGFIRE_TOKEN"):
        print("⚠️  LOGFIRE_TOKEN not set. Logging will be local only.")
        print("   Set LOGFIRE_TOKEN environment variable to send logs to Logfire.")
    
    asyncio.run(test_logging()) 