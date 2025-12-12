# Development Documentation

## Project Overview

This is a production-ready agentic application for image and video generation using fal.ai models. The project follows best practices for Python development with proper code organization, testing, and documentation.

## Architecture

### Directory Structure

```
.
├── src/                        # Main application source code
│   ├── __init__.py
│   ├── agents/                 # Agent implementations
│   │   ├── __init__.py
│   │   ├── smolagent.py       # Smolagents framework (recommended)
│   │   └── legacy_agent.py    # Legacy LLM-based intent detection
│   ├── tools/                  # Fal.ai tool implementations
│   │   ├── __init__.py
│   │   └── fal_tools.py       # Image/video generation tools
│   ├── ui/                     # User interface components
│   │   ├── __init__.py
│   │   └── gradio_interface.py
│   └── core/                   # Core utilities
│       ├── __init__.py
│       └── utils.py           # Helper functions
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py            # Application settings
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_utils.py
│   └── test_config.py
├── main_smolagent.py          # Entry point for smolagents version
├── main_legacy.py             # Entry point for legacy version
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
├── pytest.ini                 # Pytest configuration
├── Makefile                   # Development commands
├── .env                       # Environment variables (not in git)
├── .gitignore                 # Git ignore rules
├── README.md                  # User documentation
└── DEVELOPMENT.md             # This file
```

## Code Organization

### Agents (`src/agents/`)

**smolagent.py**: Recommended implementation using the smolagents framework
- Automatically selects tools based on user input
- Streams agent thinking process
- Handles multimodal input/output

**legacy_agent.py**: Legacy implementation using fal-ai/any-llm
- Manual intent detection
- Simpler but less flexible
- Good for understanding the basics

### Tools (`src/tools/`)

**fal_tools.py**: Implements three main tools
- `FalImageGenerationTool`: Generate images from text
- `FalVideoGenerationTool`: Create videos from text
- `FalImageEditTool`: Edit existing images

Each tool follows the smolagents `Tool` interface with:
- Descriptive name and documentation
- Typed inputs with nullable options
- Error handling
- Output formatting

### UI (`src/ui/`)

**gradio_interface.py**: Gradio interface for legacy agent
- Simple chat interface
- Image upload support
- Example prompts

### Core (`src/core/`)

**utils.py**: Utility functions for agent workflow
- `parse_image_paths()`: Extract file paths from text
- `pull_message_from_step()`: Format agent step logs
- `stream_from_smolagent()`: Stream agent execution

### Configuration (`config/`)

**settings.py**: Centralized configuration
- Environment variable loading
- Default values for all settings
- Validation logic
- Type hints for all settings

## Development Setup

### 1. Clone and Install

```bash
git clone <repo-url>
cd "Magic Hour ML role"
make install-dev
```

### 2. Environment Variables

Create `.env`:
```env
FAL_KEY=your_fal_api_key
HF_TOKEN=your_huggingface_token
```

### 3. Run Application

```bash
# Smolagents version (recommended)
make run-smolagent

# Legacy version
make run-legacy
```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style

3. **Format code**
   ```bash
   make format
   ```

4. **Run linting**
   ```bash
   make lint
   ```

5. **Run tests**
   ```bash
   make test
   ```

6. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```

### Code Style

- Follow PEP 8
- Use type hints for all function parameters and return values
- Write docstrings in Google style
- Keep functions under 50 lines when possible
- Use meaningful variable names

### Example Function

```python
def process_image(
    image_path: str,
    model: str = "nano-banana",
    width: int = 720
) -> Tuple[Image.Image, str]:
    """Process an image using the specified model
    
    Args:
        image_path: Path to the input image
        model: Model identifier to use
        width: Desired output width
        
    Returns:
        Tuple of (processed_image, status_message)
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If model is not supported
    """
    # Implementation
    pass
```

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_tools.py -v

# Specific test
pytest tests/test_tools.py::TestFalImageGenerationTool::test_init -v
```

### Writing Tests

Place tests in `tests/` directory matching the module structure:

```python
"""Tests for fal_tools module"""
import pytest
from src.tools.fal_tools import FalImageGenerationTool


class TestFalImageGenerationTool:
    """Tests for FalImageGenerationTool"""
    
    def test_initialization(self):
        """Test tool is initialized correctly"""
        tool = FalImageGenerationTool()
        assert tool.name == "fal_image_generation"
    
    @pytest.mark.integration
    def test_generation(self):
        """Test actual image generation"""
        # This would call the actual API
        pass
```

## Adding New Features

### Adding a New Tool

1. Create tool class in `src/tools/fal_tools.py`:

```python
class FalNewTool(Tool):
    name = "fal_new_tool"
    description = "Description of what this tool does"
    inputs = {
        "param1": {
            "type": "string",
            "description": "Description"
        }
    }
    output_type = "string"
    
    def forward(self, param1: str) -> str:
        """Implementation"""
        pass
```

2. Add to agent's tool list in `src/agents/smolagent.py`:

```python
self.tools = [
    FalImageGenerationTool(),
    FalNewTool(),  # Add here
]
```

3. Write tests in `tests/test_tools.py`

4. Update documentation

### Adding Configuration Options

1. Add to `config/settings.py`:

```python
class Settings:
    NEW_SETTING: str = "default_value"
```

2. Use in code:

```python
from config.settings import settings

value = settings.NEW_SETTING
```

## Debugging

### Enable Debug Mode

In `main_smolagent.py` or `main_legacy.py`:

```python
demo.launch(
    debug=True,  # Already enabled
    show_error=True
)
```

### Check Agent Logs

Agent logs are printed to console:

```python
print(f"Agent output: {agent_output}")
```

### Inspect Tool Calls

Tools log their inputs and outputs:

```python
print(f"Tool called with: {args}")
print(f"Tool returned: {result}")
```

## Deployment

### Production Checklist

- [ ] Set appropriate API rate limits
- [ ] Configure proper error handling
- [ ] Set up logging infrastructure
- [ ] Add monitoring/alerting
- [ ] Use production-grade web server
- [ ] Secure API keys (use secrets management)
- [ ] Add authentication if needed
- [ ] Test with production data
- [ ] Document deployment process
- [ ] Set up CI/CD pipeline

### Docker Deployment (TODO)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main_smolagent.py"]
```

## Performance Considerations

- **Caching**: Consider caching agent responses
- **Rate Limiting**: Implement rate limiting for API calls
- **Async**: Use async operations for I/O-bound tasks
- **Batch Processing**: Process multiple requests together when possible
- **Resource Limits**: Set max image sizes, video durations, etc.

## Troubleshooting

### Common Issues

1. **Import Errors**: Check PYTHONPATH and package installation
2. **API Key Issues**: Verify `.env` file exists and is loaded
3. **Port Conflicts**: Change `SERVER_PORT` in settings
4. **Memory Issues**: Reduce batch sizes, image dimensions
5. **Timeout Errors**: Increase timeout values, check API status

## Contributing

See main README.md for contribution guidelines.

## Resources

- [Fal.ai API Docs](https://fal.ai/docs)
- [Smolagents Docs](https://huggingface.co/docs/smolagents)
- [Gradio Docs](https://www.gradio.app/docs)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
