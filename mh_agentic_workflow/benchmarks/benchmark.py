"""
Fixed benchmark suite for fal.ai agentic workflow
"""
import time
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
import statistics


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    test_name: str
    category: str
    prompt: str
    expected_tool: str | None
    actual_tool: str | None = None
    tool_correct: bool = False
    params_correct: bool = False
    completed: bool = False
    error: str | None = None
    
    # Timing (seconds)
    ttft: float = 0.0
    tool_selection_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    
    # Resource usage
    llm_tokens_used: int = 0
    api_calls_made: int = 0
    retries: int = 0


@dataclass 
class BenchmarkSuite:
    """Aggregated benchmark results"""
    results: list[BenchmarkResult] = field(default_factory=list)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def summary(self) -> dict:
        if not self.results:
            return {}
        
        completed = [r for r in self.results if r.completed]
        tool_correct = [r for r in self.results if r.tool_correct]
        
        times = [r.total_time for r in completed]
        
        return {
            "total_tests": len(self.results),
            "success_rate": len(completed) / len(self.results) * 100,
            "tool_accuracy": len(tool_correct) / len(self.results) * 100,
            "avg_latency_s": statistics.mean(times) if times else 0,
            "p50_latency_s": statistics.median(times) if times else 0,
            "p95_latency_s": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else 0,
            "by_category": self._by_category()
        }
    
    def _by_category(self) -> dict:
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"total": 0, "correct": 0, "completed": 0}
            categories[r.category]["total"] += 1
            if r.tool_correct:
                categories[r.category]["correct"] += 1
            if r.completed:
                categories[r.category]["completed"] += 1
        return categories


# =============================================================================
# TEST CASES
# =============================================================================

TOOL_SELECTION_TESTS = [
    # Basic Image Generation
    {
        "category": "image_gen_basic",
        "prompt": "Generate an image of a sunset over mountains",
        "expected_tool": "fal_image_generation",
        "expected_params": {"num_images": 1}
    },
    {
        "category": "image_gen_basic",
        "prompt": "Create a picture of a cat sitting on a windowsill",
        "expected_tool": "fal_image_generation",
        "expected_params": {}
    },
    {
        "category": "image_gen_basic",
        "prompt": "Make me an image of a futuristic city",
        "expected_tool": "fal_image_generation",
        "expected_params": {}
    },
    
    # Image Generation with Parameters
    {
        "category": "image_gen_params",
        "prompt": "Generate 3 images of a dragon in portrait format",
        "expected_tool": "fal_image_generation",
        "expected_params": {"num_images": 3}
    },
    {
        "category": "image_gen_params",
        "prompt": "Create a wide landscape image of a beach, 1920x1080",
        "expected_tool": "fal_image_generation",
        "expected_params": {"width": 1920, "height": 1080}
    },
    {
        "category": "image_gen_params",
        "prompt": "Generate a square 512x512 image of a robot",
        "expected_tool": "fal_image_generation",
        "expected_params": {"width": 512, "height": 512}
    },
    
    # Model Selection
    {
        "category": "model_selection",
        "prompt": "Generate a photorealistic portrait using pro mode",
        "expected_tool": "fal_image_generation",
        "expected_params": {"mode": "pro"}
    },
    {
        "category": "model_selection",
        "prompt": "Quick sketch of a house, use fast mode",
        "expected_tool": "fal_image_generation",
        "expected_params": {"mode": "fast"}
    },
    {
        "category": "model_selection",
        "prompt": "Create an anime style character",
        "expected_tool": "fal_image_generation",
        "expected_params": {}
    },
    {
        "category": "model_selection",
        "prompt": "Generate a UI mockup for a mobile app",
        "expected_tool": "fal_image_generation",
        "expected_params": {}
    },
    
    # Image Editing
    {
        "category": "image_edit",
        "prompt": "Make this image brighter",
        "expected_tool": "fal_image_edit",
        "expected_params": {},
        "requires_image": True
    },
    {
        "category": "image_edit",
        "prompt": "Remove the background from my uploaded photo",
        "expected_tool": "fal_image_edit",
        "expected_params": {},
        "requires_image": True
    },
    {
        "category": "image_edit",
        "prompt": "Change the color of the car to red in my image",
        "expected_tool": "fal_image_edit",
        "expected_params": {},
        "requires_image": True
    },
    {
        "category": "image_edit",
        "prompt": "Add a hat to the person in this photo",
        "expected_tool": "fal_image_edit",
        "expected_params": {},
        "requires_image": True
    },
    
    # # Video Generation
    # {
    #     "category": "video_gen",
    #     "prompt": "Create a 5 second video of waves crashing",
    #     "expected_tool": "fal_video_generation",
    #     "expected_params": {}
    # },
    # {
    #     "category": "video_gen",
    #     "prompt": "Generate an animation of a butterfly flying",
    #     "expected_tool": "fal_video_generation",
    #     "expected_params": {}
    # },
    
    # Edge Cases
    {
        "category": "edge_case",
        "prompt": "I want a picture",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Vague request - should still route to image gen"
    },
    {
        "category": "edge_case",
        "prompt": "Transform my photo into a painting style",
        "expected_tool": "fal_image_edit",
        "expected_params": {},
        "requires_image": True,
        "notes": "Style transfer is editing"
    },
    {
        "category": "edge_case",
        "prompt": "Generate an image and then make it darker",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Multi-step - should start with generation"
    },
    {
        "category": "edge_case",
        "prompt": "What models do you support?",
        "expected_tool": None,  # No tool expected - just a question
        "expected_params": {},
        "notes": "Question, not a generation request"
    },
    
    # Robustness
    {
        "category": "robustness",
        "prompt": "generate IMAGE of cat",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Mixed case"
    },
    {
        "category": "robustness",
        "prompt": "   create an image of a dog   ",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Extra whitespace"
    },
    {
        "category": "robustness",
        "prompt": "Generate an image\n\nof a forest",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Newlines in prompt"
    },
    {
        "category": "robustness",
        "prompt": "Make 10000 images of cats",
        "expected_tool": "fal_image_generation",
        "expected_params": {},
        "notes": "Should cap at max"
    },
]

LATENCY_TESTS = [
    {
        "category": "latency_simple",
        "prompt": "Generate an image of a red apple",
        "expected_tool": "fal_image_generation",
        "max_tool_selection_ms": 2000,
        "max_total_ms": 10000
    },
    {
        "category": "latency_simple",
        "prompt": "Create a blue square",
        "expected_tool": "fal_image_generation",
        "max_tool_selection_ms": 2000,
        "max_total_ms": 10000
    },
    {
        "category": "latency_complex",
        "prompt": "Generate 4 images of a medieval castle at sunset with dramatic lighting, use pro mode, 1024x768 landscape format",
        "expected_tool": "fal_image_generation",
        "max_tool_selection_ms": 3000,
        "max_total_ms": 45000
    },
]


class AgentBenchmark:
    """Run benchmarks against the agent"""
    
    # Known tool names that might appear in agent code
    TOOL_PATTERNS = {
        "fal_image_generation": [
            "fal_image_generation",
            "FalImageGenerationTool",
            "image_generation",
        ],
        "fal_video_generation": [
            "fal_video_generation", 
            "FalVideoGenerationTool",
            "video_generation",
        ],
        "fal_image_edit": [
            "fal_image_edit",
            "FalImageEditTool", 
            "image_edit",
        ],
    }
    
    def __init__(self, agent_app):
        self.agent = agent_app
        self.suite = BenchmarkSuite()
    
    def _extract_tool_from_code(self, code: str) -> str | None:
        """Extract tool name from generated code"""
        for canonical_name, patterns in self.TOOL_PATTERNS.items():
            for pattern in patterns:
                if pattern in code:
                    return canonical_name
        return None
    
    def _extract_tool_from_step(self, step) -> tuple[str | None, dict]:
        """Extract tool name and params from an agent step"""
        tool_name = None
        params = {}
        
        # 1. Inspect the Tool Call
        if hasattr(step, 'tool_calls') and step.tool_calls:
            tool_call = step.tool_calls[0]
            raw_name = getattr(tool_call, 'name', None)
            
            # 2. THE FIX: Handle Python Interpreter wrappers
            # If the tool is the interpreter, we must look INSIDE the code arguments
            if raw_name == "python_interpreter":
                # The code is usually passed as the argument to the interpreter
                code_content = str(tool_call.arguments)
                
                # Use your existing regex helper to find 'fal_image_generation' inside the code
                extracted_tool = self._extract_tool_from_code(code_content)
                
                if extracted_tool:
                    tool_name = extracted_tool
                    params = self._extract_params_from_code(code_content)
                else:
                    # Fallback: If we can't find a known tool pattern in the code,
                    # we log it as python_interpreter (which will likely fail the test, as correct)
                    tool_name = raw_name
            else:
                # Standard tool call (not code interpreter)
                tool_name = raw_name
                if hasattr(tool_call, 'arguments'):
                    params = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        
        # 3. Fallback: Parse from code execution if tool_calls didn't work or wasn't present
        if tool_name is None:
            code = None
            if hasattr(step, 'llm_output'):
                code = step.llm_output
            elif hasattr(step, 'code'):
                code = step.code
            
            if code:
                tool_name = self._extract_tool_from_code(str(code))
                params = self._extract_params_from_code(str(code))
        
        return tool_name, params
    
    def _extract_params_from_code(self, code: str) -> dict:
        """Extract parameters from generated code"""
        params = {}
        
        # Extract num_images
        match = re.search(r'num_images\s*=\s*(\d+)', code)
        if match:
            params['num_images'] = int(match.group(1))
        
        # Extract width
        match = re.search(r'width\s*=\s*(\d+)', code)
        if match:
            params['width'] = int(match.group(1))
        
        # Extract height
        match = re.search(r'height\s*=\s*(\d+)', code)
        if match:
            params['height'] = int(match.group(1))
        
        # Extract mode
        match = re.search(r'mode\s*=\s*["\'](\w+)["\']', code)
        if match:
            params['mode'] = match.group(1)
        
        # Extract duration (for video)
        match = re.search(r'duration\s*=\s*(\d+)', code)
        if match:
            params['duration'] = int(match.group(1))
        
        return params
    
    def run_tool_selection_tests(self, tests: list[dict] = None) -> BenchmarkSuite:
        """Run tool selection accuracy tests"""
        tests = tests or TOOL_SELECTION_TESTS
        
        for test in tests:
            result = BenchmarkResult(
                test_name=f"tool_selection_{test['category']}",
                category=test["category"],
                prompt=test["prompt"],
                expected_tool=test.get("expected_tool")
            )
            
            start = time.perf_counter()
            
            try:
                tool_selected, params, output, completed = self._run_single(
                    test["prompt"],
                    has_image=test.get("requires_image", False)
                )
                
                result.total_time = time.perf_counter() - start
                result.actual_tool = tool_selected
                result.completed = completed
                
                # Handle None expected_tool (questions/non-tool requests)
                if test.get("expected_tool") is None:
                    # For questions, success = no tool was called OR just text response
                    result.tool_correct = (tool_selected is None)
                else:
                    result.tool_correct = (tool_selected == test["expected_tool"])
                
                # Check parameter accuracy
                if test.get("expected_params") and params:
                    result.params_correct = self._check_params(
                        params, test["expected_params"]
                    )
                else:
                    result.params_correct = True
                    
            except Exception as e:
                result.error = str(e)
                result.total_time = time.perf_counter() - start
            
            self.suite.add(result)
            
            # Better output formatting
            status = "✓" if result.tool_correct else "✗"
            expected = test.get("expected_tool") or "None"
            actual = result.actual_tool or "None"
            print(f"{status} {test['prompt'][:50]}...")
            if not result.tool_correct:
                print(f"    Expected: {expected}, Got: {actual}")
        
        return self.suite
    
    def run_latency_tests(self, tests: list[dict] = None, iterations: int = 3):
        """Run latency benchmarks with multiple iterations"""
        tests = tests or LATENCY_TESTS
        
        for test in tests:
            latencies = []
            
            for i in range(iterations):
                result = BenchmarkResult(
                    test_name=f"latency_{test['category']}_{i}",
                    category=test["category"],
                    prompt=test["prompt"],
                    expected_tool=test["expected_tool"]
                )
                
                start = time.perf_counter()
                
                try:
                    tool_selected, params, output, completed = self._run_single(test["prompt"])
                    result.total_time = time.perf_counter() - start
                    result.completed = completed
                    result.actual_tool = tool_selected
                    result.tool_correct = (tool_selected == test["expected_tool"])
                    latencies.append(result.total_time)
                except Exception as e:
                    result.error = str(e)
                
                self.suite.add(result)
            
            if latencies:
                avg = statistics.mean(latencies)
                max_allowed = test.get("max_total_ms", 30000) / 1000
                status = "✓" if avg < max_allowed else "✗"
                print(f"{status} {test['prompt'][:40]}... avg={avg:.2f}s (max={max_allowed}s)")
        
        return self.suite
    
    def _run_single(self, prompt: str, has_image: bool = False) -> tuple[str | None, dict, str | None, bool]:
        """Run a single agent request and extract tool info
        
        Returns:
            tuple: (tool_selected, params, final_output, completed)
        """
        image_context = ""
        if has_image:
            image_context = "\n\nUploaded file path: /tmp/test_image.png"
        
        full_prompt = prompt + image_context
        
        tool_selected = None
        params = {}
        final_output = None
        completed = False
        
        try:
            # Run with streaming to capture steps
            for step in self.agent.agent.run(full_prompt, stream=True):
                # Try to extract tool from this step
                step_tool, step_params = self._extract_tool_from_step(step)
                
                if step_tool and tool_selected is None:
                    tool_selected = step_tool
                    params = step_params
                
                # Capture observations
                if hasattr(step, 'observations') and step.observations:
                    final_output = str(step.observations)
            
            completed = True
            
        except Exception as e:
            final_output = f"Error: {str(e)}"
        
        return tool_selected, params, final_output, completed
    
    def _check_params(self, actual: dict, expected: dict) -> bool:
        """Check if extracted params match expected"""
        for key, value in expected.items():
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True
    
    def export_results(self, path: str = "benchmark_results.json"):
        """Export results to JSON"""
        output = {
            "summary": self.suite.summary(),
            "results": [asdict(r) for r in self.suite.results]
        }
        Path(path).write_text(json.dumps(output, indent=2))
        print(f"Results exported to {path}")


def run_benchmarks():
    """Main entry point for running benchmarks"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.agents.smolagent import SmolagentFalApp
    from config.settings import settings
    
    # Initialize agent
    print("Initializing agent...")
    app = SmolagentFalApp(
        hf_token=settings.HF_TOKEN,
        fal_model_name=settings.FAL_MODEL_NAME
    )
    
    # Run benchmarks
    benchmark = AgentBenchmark(app)
    
    print("\n" + "="*60)
    print("TOOL SELECTION ACCURACY TESTS")
    print("="*60)
    benchmark.run_tool_selection_tests()
    
    print("\n" + "="*60)
    print("LATENCY TESTS")
    print("="*60)
    benchmark.run_latency_tests(iterations=3)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = benchmark.suite.summary()
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Tool Selection Accuracy: {summary['tool_accuracy']:.1f}%")
    print(f"Average Latency: {summary['avg_latency_s']:.2f}s")
    print(f"P95 Latency: {summary['p95_latency_s']:.2f}s")
    
    print("\nBy Category:")
    for cat, stats in summary['by_category'].items():
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {cat}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Export
    benchmark.export_results()
    
    return benchmark.suite


if __name__ == "__main__":
    run_benchmarks()