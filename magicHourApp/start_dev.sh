#!/bin/bash
# Magic Hour AI - Development Server Launcher (LangGraph Version)
# Starts both the Python FastAPI backend and React frontend

echo "🌟 Starting Magic Hour AI (LangGraph)..."
echo ""

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Please run this script from the magicHourApp directory"
    exit 1
fi

# Check for LangGraph workflow directory
LANGGRAPH_DIR="../mh_langgraph_workflow"
if [ ! -d "$LANGGRAPH_DIR" ]; then
    echo "❌ Error: mh_langgraph_workflow directory not found"
    exit 1
fi

# Check for .env file in LangGraph workflow
if [ ! -f "$LANGGRAPH_DIR/.env" ]; then
    echo "⚠️  No .env file found in mh_langgraph_workflow"

    # Try to copy from mh_agentic_workflow if it exists
    if [ -f "../mh_agentic_workflow/.env" ]; then
        echo "📋 Copying .env from mh_agentic_workflow..."
        cp "../mh_agentic_workflow/.env" "$LANGGRAPH_DIR/.env"
        echo "✅ .env copied successfully"
    else
        echo "❌ Please create $LANGGRAPH_DIR/.env with FAL_KEY and PRODIA_KEY"
        exit 1
    fi
fi

# Install LangGraph dependencies if requirements changed
echo "📦 Checking LangGraph dependencies..."
pip install -q -r "$LANGGRAPH_DIR/requirements.txt"

# Start backend in background
echo "🚀 Starting FastAPI backend (LangGraph) on http://localhost:8000..."
python api_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "🎨 Starting React frontend on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are running!"
echo "   📡 Backend API: http://localhost:8000"
echo "   🎨 Frontend UI: http://localhost:5173"
echo "   🤖 LLM Model: ${FAL_MODEL_NAME:-google/gemini-2.5-flash}"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 Servers stopped'; exit 0" INT

# Wait for either process to exit
wait
