# Quick Reference Guide

## Common Commands

```bash
# Installation
make install              # Install dependencies

# Running
make run-smolagent        # Run smolagents version (recommended)
make run-legacy           # Run legacy version

# Development
make clean               # Clean cache files
```

## Project Structure at a Glance

```
src/
├── agents/        # Agent implementations (brain)
├── tools/         # Fal.ai tools (actions)
├── ui/            # Gradio interfaces (presentation)
└── core/          # Utilities (helpers)

config/            # Settings and configuration
tests/             # Test suite
main_*.py          # Entry points
```

## Common Tasks

### Add a New Fal.ai Tool

1. Edit `src/tools/fal_tools.py`
2. Create class extending `Tool`
3. Add to agent in `src/agents/smolagent.py`
4. Write tests in `tests/test_tools.py`

### Change Configuration

Edit `config/settings.py`:
```python
class Settings:
    YOUR_SETTING = "value"
```

Use it:
```python
from config.settings import settings
value = settings.YOUR_SETTING
```

### Add Dependencies

1. Add to `requirements.txt`
2. Run `make install`
3. Update `pyproject.toml` if needed

## Import Patterns

```python
# Tools
from src.tools.fal_tools import FalImageGenerationTool

# Agents
from src.agents.smolagent import SmolagentFalApp

# Utils
from src.core.utils import parse_image_paths

# Config
from config.settings import settings

# UI
from src.ui.gradio_interface import GradioInterface
```

## Environment Variables

Create `.env` file:
```env
FAL_KEY=your_fal_api_key
HF_TOKEN=your_huggingface_token
```

## File Organization Rules

| What | Where | Example |
|------|-------|---------|
| Agent logic | `src/agents/` | New agent class |
| Fal.ai tools | `src/tools/` | New generation tool |
| UI components | `src/ui/` | Custom Gradio interface |
| Utilities | `src/core/` | Helper functions |
| Configuration | `config/` | Settings, constants |
| Documentation | Root | README, guides |

## Debugging Tips

1. **Enable debug mode**: Already on in entry points
2. **Check logs**: Look in terminal output
3. **Print agent logs**: `print(agent.logs)`
4. **Use breakpoints**: Add `import pdb; pdb.set_trace()`
5. **Check config**: `print(settings.YOUR_SETTING)`

## Git Workflow

```bash
# Start new feature
git checkout -b feature/your-feature

# Make changes, then:
git add .
git commit -m "Add feature X"
git push origin feature/your-feature
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run `make install` |
| Port in use | Change `SERVER_PORT` in `config/settings.py` |
| API errors | Check `.env` file has valid keys |
| Module not found | Check you're in project root |

## Documentation

- `README.md` - User guide, quick start
- `DEVELOPMENT.md` - Developer guide, architecture
- `MIGRATION.md` - Migration from old structure
- `QUICKREF.md` - This file
- `AGENT_WORKFLOW_GUIDE.md` - Original guide

## Code Style

| File | Purpose | Use When |
|------|---------|----------|
| `main_smolagent.py` | Smolagents app | Full agentic workflow |
| `main_legacy.py` | Legacy app | Simple intent detection |

## Configuration Options

See `config/settings.py`:
- `FAL_KEY` - Fal.ai API key
- `HF_TOKEN` - Hugging Face token
- `LLM_MODEL_ID` - Model to use
- `MAX_AGENT_STEPS` - Max iterations
- `SERVER_PORT` - Web server port
- More...

## Resources

- Fal.ai: https://fal.ai/docs
- Smolagents: https://huggingface.co/docs/smolagents
- Gradio: https://www.gradio.app/docs

---

**Quick Start:**
```bash
make install
# Add API keys to .env
make run-smolagent
```
