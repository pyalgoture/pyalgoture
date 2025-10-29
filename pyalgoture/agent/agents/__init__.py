import json
from abc import abstractmethod
from collections.abc import Sequence
from typing import Annotated, Any

from typing_extensions import TypedDict

from ...utils.logger import get_logger
from ...utils.models import CommonResponseSchema

logger = get_logger()


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    data: Annotated[dict[str, Any], merge_dicts]
    analysis: Annotated[dict[str, Any], merge_dicts]
    decision: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]


class BaseAgent:
    live_only: bool = False

    @abstractmethod
    def __call__(self, state: AgentState) -> CommonResponseSchema:
        raise NotImplementedError("Subclasses must implement __call__")


def show_agent_reasoning(output: Any, agent_name: str) -> None:
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, int | float | bool | str):
            return obj
        elif isinstance(obj, list | tuple):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, dict | list):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)
