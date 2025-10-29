from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from ...utils.logger import get_logger
from .api_schemas import BgJobStarted
from .webserver_bgwork import ApiBG

logger = get_logger()


class OperationalException(Exception):
    """
    Requires manual intervention and will stop the bot.
    Most of the time, this is caused by an invalid Configuration.
    """


# Private API, protected by authentication and webserver_mode dependency
router = APIRouter(tags=["reconciliation", "webserver"])


def __run_reconciliation(job_id: str, config_loc: dict):
    try:
        ApiBG.jobs[job_id]["is_running"] = True

        # with FtNoDBContext():
        #     exchange = get_exchange(config_loc)

        #     def ft_callback(task) -> None:
        #         ApiBG.jobs[job_id]["progress_tasks"][str(task.id)] = {
        #             "progress": task.completed,
        #             "total": task.total,
        #             "description": task.description,
        #         }

        #     pt = get_progress_tracker(ft_callback=ft_callback)

        #     download_data(config_loc, exchange, progress_tracker=pt)
        #     ApiBG.jobs[job_id]["status"] = "success"

        ApiBG.jobs[job_id]["status"] = "success"
    except (OperationalException, Exception) as e:
        logger.exception(e)
        ApiBG.jobs[job_id]["error"] = str(e)
        ApiBG.jobs[job_id]["status"] = "failed"
    finally:
        ApiBG.jobs[job_id]["is_running"] = False
        ApiBG.reconciliation_running = False


@router.post("/reconciliation", response_model=BgJobStarted)
def reconciliation(background_tasks: BackgroundTasks):
    if ApiBG.reconciliation_running:
        raise HTTPException(status_code=400, detail="Reconciliation is already running.")

    job_id = ApiBG.get_job_id()

    ApiBG.jobs[job_id] = {
        "category": "reconciliation",
        "status": "pending",
        "progress": None,
        "progress_tasks": {},
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_reconciliation, job_id)
    ApiBG.reconciliation_running = True

    return {
        "status": "Reconciliation started in background.",
        "job_id": job_id,
    }
