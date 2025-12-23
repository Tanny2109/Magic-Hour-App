#!/bin/bash

# Production startup script for Magic Hour AI (LangGraph Version)

echo "🚀 Building Magic Hour AI for production..."

cd "$(dirname "$0")"

# Check for LangGraph workflow directory
LANGGRAPH_DIR="../mh_langgraph_workflow"
if [ ! -d "$LANGGRAPH_DIR" ]; then
    echo "❌ Error: mh_langgraph_workflow directory not found"
    exit 1
fi

# Check for .env file
if [ ! -f "$LANGGRAPH_DIR/.env" ]; then
    echo "⚠️  No .env file found in mh_langgraph_workflow"
    if [ -f "../mh_agentic_workflow/.env" ]; then
        echo "📋 Copying .env from mh_agentic_workflow..."
        cp "../mh_agentic_workflow/.env" "$LANGGRAPH_DIR/.env"
    else
        echo "❌ Please create $LANGGRAPH_DIR/.env with FAL_KEY and PRODIA_KEY"
        exit 1
    fi
fi

# Build frontend
echo "📦 Building React frontend..."
npm install
npm run build

# Check if build succeeded
if [ ! -d "dist" ]; then
    echo "❌ Frontend build failed - dist/ directory not found"
    exit 1
fi

echo "✅ Frontend built successfully"

# Install Python dependencies (both local and LangGraph)
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
pip install -r "$LANGGRAPH_DIR/requirements.txt"

# Set production mode
export PRODUCTION=true

# Start server
echo "🌟 Starting Magic Hour AI (LangGraph) in production mode..."
echo "📡 Server will be available at http://0.0.0.0:8000"
echo "🤖 LLM Model: ${FAL_MODEL_NAME:-google/gemini-2.5-flash}"
python api_server.py
