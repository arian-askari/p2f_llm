"""
Terminal-first Function Factory runner.

Usage:
    python func_factory_cli.py --config config.yaml
    python func_factory_cli.py --config config.json

Config supports YAML or JSON.
Dataset supports JSON or JSONL.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import asyncio
import logging
import warnings
import subprocess
import tempfile
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class EvalItem(BaseModel):
    obj: Any
    ground_truth: Any


class CodeGenerationRequest(BaseModel):
    prompt_text: str
    eval_dataset: list[EvalItem]
    func_file_name: str
    func_base_path_dir: str = "./built_functions/"
    python_command: str = "python3"
    generated_func_return_type: str = "bool | dict"

    app_name: str = "function_factory_app"
    user_id: str = "user_1"
    session_id: str = "session_001"
    model_name: str = "gemini-2.5-flash"
    query: str = "generate code"


class CodeGenerationRunResult(BaseModel):
    final_response_text: str
    eval_result: dict[str, Any] = Field(default_factory=dict)
    session_state: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------------------
# File loading
# ------------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for YAML config files. Install with: pip install pyyaml"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_config(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return load_yaml(config_path)
    if suffix == ".json":
        return load_json(config_path)
    raise ValueError(f"Unsupported config file type: {config_path}")


def load_dataset(dataset_path: Path) -> list[EvalItem]:
    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        items: list[EvalItem] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(EvalItem.model_validate(json.loads(line)))
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSONL dataset row at line {line_no}: {e}"
                    ) from e
        return items

    if suffix == ".json":
        raw = load_json(dataset_path)
        if not isinstance(raw, list):
            raise ValueError("JSON dataset must be a list of EvalItem objects")
        return [EvalItem.model_validate(item) for item in raw]

    raise ValueError(
        f"Unsupported dataset file type: {dataset_path}. Use .json or .jsonl"
    )


# ------------------------------------------------------------------------------
# Tool state
# ------------------------------------------------------------------------------
def build_tool_state(request: CodeGenerationRequest) -> dict[str, Any]:
    return {
        "eval_dataset": request.eval_dataset,
        "func_base_path_dir": request.func_base_path_dir,
        "python_command": request.python_command,
        "func_file_name": request.func_file_name,
        "func_name": request.func_file_name,
        "prompt_text": request.prompt_text,
        "generated_func_return_type": request.generated_func_return_type,
        "eval_result": {},
    }


# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------
def execute_python_code(code_text: str, python_command: str = "python3") -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(code_text)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            [python_command, temp_file_path],
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def evaluate_generated_function(tool_context: ToolContext) -> dict[str, Any]:
    eval_dataset = tool_context.state.get("eval_dataset", [])
    func_base_path_dir = tool_context.state.get("func_base_path_dir", "./built_functions/")
    func_file_name = tool_context.state.get("func_file_name", "")
    func_name = tool_context.state.get("func_name", "")

    if not func_file_name:
        raise ValueError("Missing 'func_file_name' in tool_context.state")
    if not func_name:
        raise ValueError("Missing 'func_name' in tool_context.state")

    if not func_file_name.endswith(".py"):
        func_file_name += ".py"

    file_path = os.path.join(func_base_path_dir, func_file_name)

    if not os.path.exists(file_path):
        result = {
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "details": [],
            "message": f"Function file not found yet: {file_path}",
        }
        tool_context.state["eval_result"] = result
        return result

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target_func = getattr(module, func_name, None)
    if target_func is None:
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'")

    total = len(eval_dataset)
    if total == 0:
        result = {
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "details": [],
            "message": "eval_dataset is empty",
        }
        tool_context.state["eval_result"] = result
        return result

    correct = 0
    details = []

    for idx, eval_item in enumerate(eval_dataset):
        reason = {}

        try:
            prediction = target_func(eval_item, reason)

            if isinstance(prediction, bool):
                normalized_prediction = "eligible" if prediction else "not_eligible"
            else:
                normalized_prediction = prediction

            ground_truth = getattr(eval_item, "ground_truth", None)
            is_correct = int(normalized_prediction == ground_truth)
            correct += is_correct

            details.append({
                "index": idx,
                "input": repr(getattr(eval_item, "obj", None)),
                "prediction": normalized_prediction,
                "ground_truth": ground_truth,
                "match": bool(is_correct),
                "reason": reason.get("reason"),
            })

        except Exception as e:
            details.append({
                "index": idx,
                "input": repr(getattr(eval_item, "obj", None)),
                "prediction": None,
                "ground_truth": getattr(eval_item, "ground_truth", None),
                "match": False,
                "reason": f"Execution error: {str(e)}",
            })

    eval_result = {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "details": details,
    }

    tool_context.state["eval_result"] = eval_result
    return eval_result


def save_python_code(
    code_text: str,
    base_path_dir: str,
    file_name: str,
    tool_context: ToolContext,
) -> str:
    os.makedirs(base_path_dir, exist_ok=True)

    if not file_name.endswith(".py"):
        file_name += ".py"

    file_path = os.path.join(base_path_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_text)

    evaluate_generated_function(tool_context)
    return file_path


# ------------------------------------------------------------------------------
# Capability
# ------------------------------------------------------------------------------
class CodeGeneratorCapability:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    def _build_agent(self) -> Agent:
        instruction = """
You are function factory agent that can convert a deterministic prompt to a python code given a data model.
If a prompt is not convertable to python code, you would let the user know by mentioning
'sorry the provided prompt cannot be converted to python code because <your reasoning here>'.

The prompt that should be converted to a function is the following:
<start_of_prompt> {prompt_text} </end_of_prompt>

The function that you generate must only expect as its input an instance of obj in EvalItem.
E.g., {func_name}(eval_item: EvalItem, return_reason: dict) -> {generated_func_return_type}

Following is an instance of eval_item: {eval_dataset[0]}

For simplicity, you can re-define the object inside EvalItem above than your generated function so your function can be executed!

Steps that you must follow:
1- Generate the python code and implement a function
2- Run it internally in your sandbox and make sure it is executable.
3- If it was runnable in your sandbox, then run it with execute_python_code tool because we need to make sure we can run it in real-world envorinment as well
4- Use save_python_code tool to save the generated code and its test cases into directory of {func_base_path_dir} and give its file name {func_file_name}.py and use same name of {func_file_name} for its function.
5- Execute your evaluate_generated_function tool without passing any param to it.
6- Print {eval_result}
        """

        return Agent(
            name="code_generator_agent",
            model=self.model_name,
            description="Reusable function factory capability for deterministic code generation.",
            instruction=instruction,
            tools=[execute_python_code, save_python_code, evaluate_generated_function],
        )

    async def _call_agent_async(
        self,
        runner: Runner,
        query: str,
        user_id: str,
        session_id: str,
    ) -> str:
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break

        return final_response_text

    async def run(self, request: CodeGenerationRequest) -> CodeGenerationRunResult:
        agent = self._build_agent()
        session_service = InMemorySessionService()

        session = await session_service.create_session(
            app_name=request.app_name,
            user_id=request.user_id,
            session_id=request.session_id,
            state=build_tool_state(request),
        )

        runner = Runner(
            agent=agent,
            app_name=request.app_name,
            session_service=session_service,
        )

        final_response_text = await self._call_agent_async(
            runner=runner,
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        return CodeGenerationRunResult(
            final_response_text=final_response_text,
            eval_result=session.state.get("eval_result", {}),
            session_state=dict(session.state),
        )


# ------------------------------------------------------------------------------
# Request builder from config
# ------------------------------------------------------------------------------
def resolve_prompt_text(config: dict[str, Any], config_path: Path) -> str:
    prompt_text = config.get("prompt_text")
    prompt_file = config.get("prompt_file")

    if prompt_text and prompt_file:
        raise ValueError("Use only one of 'prompt_text' or 'prompt_file' in config")

    if prompt_text:
        return prompt_text

    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (config_path.parent / prompt_path).resolve()
        return load_text(prompt_path)

    raise ValueError("Config must contain either 'prompt_text' or 'prompt_file'")


def build_request_from_config(config_path: Path) -> CodeGenerationRequest:
    config = load_config(config_path)

    dataset_file = config.get("dataset_file")
    if not dataset_file:
        raise ValueError("Config must contain 'dataset_file'")

    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()

    eval_dataset = load_dataset(dataset_path)
    prompt_text = resolve_prompt_text(config, config_path)

    return CodeGenerationRequest(
        prompt_text=prompt_text,
        eval_dataset=eval_dataset,
        func_file_name=config["func_file_name"],
        func_base_path_dir=config.get("func_base_path_dir", "./built_functions/"),
        python_command=config.get("python_command", "python3"),
        generated_func_return_type=config.get("generated_func_return_type", "bool | dict"),
        app_name=config.get("app_name", "function_factory_app"),
        user_id=config.get("user_id", "user_1"),
        session_id=config.get("session_id", "session_001"),
        model_name=config.get("model_name", "gemini-2.5-flash"),
        query=config.get("query", "generate code"),
    )


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the function factory capability from terminal using YAML/JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (.yaml, .yml, or .json)",
    )
    parser.add_argument(
        "--print-session-state",
        action="store_true",
        help="Print the final session state as JSON",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    request = build_request_from_config(config_path)

    # Environment can also be supplied through shell instead of config.
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export it in your shell before running the script."
        )

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False")

    capability = CodeGeneratorCapability(model_name=request.model_name)
    result = await capability.run(request)

    print("\n=== FINAL RESPONSE ===")
    print(result.final_response_text)

    print("\n=== EVAL RESULT ===")
    print(json.dumps(result.eval_result, indent=2, ensure_ascii=False, default=str))

    if args.print_session_state:
        print("\n=== SESSION STATE ===")
        print(json.dumps(result.session_state, indent=2, ensure_ascii=False, default=str))

    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
