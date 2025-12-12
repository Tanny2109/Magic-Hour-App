#!/bin/bash
# Magic Hour AI - Development Server Launcher
# Starts both the Python FastAPI backend and React frontend

echo "🌟 Starting Magic Hour AI..."
echo ""

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Please run this script from the magicHourApp directory"
    exit 1
fi

# Start backend in background
echo "🚀 Starting FastAPI backend on http://localhost:8000..."
python api_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "🎨 Starting React frontend on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are running!"
echo "   📡 Backend API: http://localhost:8000"
echo "   🎨 Frontend UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 Servers stopped'; exit 0" INT

# Wait for either process to exit
wait
