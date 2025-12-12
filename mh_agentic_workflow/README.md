# Fal.ai Agentic Workflow

An agentic application for generating images and videos using fal.ai models with intelligent tool selection and streaming capabilities.

## 🌟 Features

- **Smolagents Framework**: LLM-powered agent that automatically selects appropriate tools and parameters
- **Multiple Fal.ai Models**: Support for nano-banana, flux-schnell, and flux-pro image generation
- **Image Editing**: Transform and enhance uploaded images
- **Streaming Interface**: Real-time display of agent thinking process
- **Gradio UI**: Beautiful, interactive web interface

## 📋 Requirements

- Python 3.8+
- Fal.ai API key
- Hugging Face API token

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
FAL_KEY=your_fal_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Run the Application

**Smolagents Version (Recommended):**
```bash
python main_smolagent.py
```

**Legacy Version:**
```bash
python main_legacy.py
```

The application will launch at `http://localhost:7860`