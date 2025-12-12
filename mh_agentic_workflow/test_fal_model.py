"""
Test script for FalAIModel with smolagents CodeAgent.

This script demonstrates how to use the custom FalAIModel with CodeAgent.
"""

import os
from smolagents import CodeAgent
from src.models import FalAIModel
from dotenv import load_dotenv
load_dotenv()


def test_basic_generation():
    """Test basic text generation with FalAIModel."""
    print("=" * 60)
    print("TEST 1: Basic Text Generation")
    print("=" * 60)

    # Initialize the model
    model = FalAIModel(
        fal_api="fal-ai/any-llm",
        fal_model_name="google/gemini-2.5-flash-lite",  # Free tier model
        temperature=0.7,
        max_tokens=1000
    )

    # Create a simple agent
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3
    )

    # Run a simple task
    result = agent.run("What is 2 + 2? Just give me the answer.")

    print(f"\nResult: {result}")
    print("\n✅ Basic generation test passed!\n")


def test_code_execution():
    """Test code execution with FalAIModel."""
    print("=" * 60)
    print("TEST 2: Code Execution")
    print("=" * 60)

    # Initialize the model
    model = FalAIModel(
        fal_api="fal-ai/any-llm",
        fal_model_name="google/gemini-2.5-flash-lite",
        temperature=0.1,
        max_tokens=2000
    )

    # Create agent
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=5
    )

    # Run a task that requires code execution
    result = agent.run("Calculate the factorial of 10 using Python code.")

    print(f"\nResult: {result}")
    print("\n✅ Code execution test passed!\n")


def test_with_streaming():
    """Test streaming generation with FalAIModel."""
    print("=" * 60)
    print("TEST 3: Streaming Generation")
    print("=" * 60)

    # Initialize the model
    model = FalAIModel(
        fal_api="fal-ai/any-llm",
        fal_model_name="google/gemini-2.5-flash-lite",
        temperature=0.7,
        max_tokens=500
    )

    # Create agent with streaming
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=3,
        stream_outputs=True
    )

    # Run with streaming
    print("\nStreaming output:")
    print("-" * 40)
    result = agent.run("Write a haiku about coding.", stream=True)

    print(f"\nFinal Result: {result}")
    print("\n✅ Streaming test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FALAI MODEL INTEGRATION TESTS")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.getenv("FAL_KEY"):
        print("⚠️  Warning: FAL_KEY environment variable not set.")
        print("Make sure you have set your fal.ai API key:")
        print("export FAL_KEY='your-api-key-here'\n")
        return

    try:
        # Run tests
        test_basic_generation()
        test_code_execution()
        test_with_streaming()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
