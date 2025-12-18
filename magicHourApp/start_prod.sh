#!/bin/bash

# Production startup script for Magic Hour AI

echo "🚀 Building Magic Hour AI for production..."

# Build frontend
echo "📦 Building React frontend..."
cd "$(dirname "$0")"
npm install
npm run build

# Check if build succeeded
if [ ! -d "dist" ]; then
    echo "❌ Frontend build failed - dist/ directory not found"
    exit 1
fi

echo "✅ Frontend built successfully"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Set production mode
export PRODUCTION=true

# Start server
echo "🌟 Starting Magic Hour AI in production mode..."
echo "📡 Server will be available at http://0.0.0.0:8000"
python api_server.py
