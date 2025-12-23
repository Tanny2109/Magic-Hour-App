# Multi-stage build for Magic Hour App (LangGraph)
FROM node:18-slim AS frontend-build

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY magicHourApp/package*.json ./
RUN npm install

# Copy all frontend source files
COPY magicHourApp/index.html ./
COPY magicHourApp/vite.config.js ./
COPY magicHourApp/eslint.config.js ./
COPY magicHourApp/src ./src

# Build frontend
RUN npm run build

# Python backend with LangGraph
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy LangGraph workflow code
COPY mh_langgraph_workflow/ ./mh_langgraph_workflow/

# Copy backend API server
COPY magicHourApp/api_server.py ./magicHourApp/
COPY magicHourApp/start_prod.sh ./magicHourApp/

# Install Python dependencies (LangGraph only)
RUN pip install --no-cache-dir -r mh_langgraph_workflow/requirements.txt

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/dist ./magicHourApp/dist

# Set environment variables
ENV PRODUCTION=true
ENV PYTHONUNBUFFERED=1

# Expose port (Render will override with PORT env var)
EXPOSE 8000

# Make start script executable
RUN chmod +x magicHourApp/start_prod.sh

# Start the FastAPI server
CMD ["python", "magicHourApp/api_server.py"]
