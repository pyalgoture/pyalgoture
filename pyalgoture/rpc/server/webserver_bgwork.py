from typing import Any, Literal
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict


class ProgressTask(TypedDict):
    progress: float
    total: float
    description: str


class JobsContainer(TypedDict):
    category: Literal["reconciliation",]
    is_running: bool
    status: str
    progress: float | None
    progress_tasks: NotRequired[dict[str, ProgressTask]]
    result: Any
    error: str | None


class ApiBG:
    # Backtesting type: Backtesting
    bt: dict[str, Any] = {
        "bt": None,
        "data": None,
        "timerange": None,
        "last_config": {},
        "bt_error": None,
    }
    # Generic background jobs

    # TODO: Change this to TTLCache
    jobs: dict[str, JobsContainer] = {}
    reconciliation_running: bool = False

    @staticmethod
    def get_job_id() -> str:
        return str(uuid4())
