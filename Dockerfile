# Multi-stage build for Magic Hour App
FROM node:18-slim AS frontend-build

WORKDIR /app/frontend
COPY magicHourApp/package*.json ./
RUN npm install
COPY magicHourApp/ ./
RUN npm run build

# Python backend
FROM python:3.11-slim

WORKDIR /app

# Copy backend code
COPY mh_agentic_workflow/ ./mh_agentic_workflow/
COPY magicHourApp/api_server.py ./magicHourApp/
COPY magicHourApp/requirements.txt ./magicHourApp/

# Install Python dependencies
RUN pip install --no-cache-dir -r mh_agentic_workflow/requirements.txt
RUN pip install --no-cache-dir -r magicHourApp/requirements.txt

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/dist ./magicHourApp/dist

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "magicHourApp/api_server.py"]
