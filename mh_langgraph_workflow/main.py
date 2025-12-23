"""Main entry point for the Magic Hour LangGraph Workflow."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def run_server():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("PRODUCTION", "").lower() != "true"
    )


def run_cli():
    """Run a simple CLI interface for testing."""
    from src.agents import create_agent

    print("Magic Hour LangGraph Agent CLI")
    print("=" * 40)
    print(f"Using LLM: {os.getenv('FAL_MODEL_NAME', 'google/gemini-2.5-flash')} via fal.ai")
    print("Commands:")
    print("  /quit - Exit the CLI")
    print("  /clear - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /images - Show generated images")
    print("=" * 40)

    agent = create_agent(
        fal_model_name=os.getenv("FAL_MODEL_NAME", "google/gemini-2.5-flash"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
    )

    thread_id = "cli-session"

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/clear":
                thread_id = f"cli-session-{os.urandom(4).hex()}"
                print("Conversation cleared.")
                continue

            if user_input.lower() == "/history":
                messages = agent.get_conversation_history(thread_id)
                print(f"\nConversation history ({len(messages)} messages):")
                for msg in messages[-10:]:  # Show last 10
                    msg_type = type(msg).__name__
                    content = msg.content[:100] if isinstance(msg.content, str) else str(msg.content)[:100]
                    print(f"  [{msg_type}]: {content}...")
                continue

            if user_input.lower() == "/images":
                paths = agent.get_generated_content(thread_id)
                print(f"\nGenerated content ({len(paths)} items):")
                for path in paths:
                    print(f"  - {path}")
                continue

            # Process the message
            print("\nAssistant: ", end="", flush=True)

            result = agent.invoke(
                message=user_input,
                thread_id=thread_id
            )

            # Extract and print response
            messages = result.get("messages", [])
            for msg in messages:
                if type(msg).__name__ == "AIMessage":
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    print(content)

            # Show any generated content
            generated = result.get("generated_content", [])
            if generated:
                print("\n[Generated content:]")
                for path in generated:
                    print(f"  - {path}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        run_server()
