from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from ...utils.logger import get_logger
from .api_schemas import BackgroundTaskStatus
from .webserver_bgwork import ApiBG

logger = get_logger()


# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


@router.get("/background", response_model=list[BackgroundTaskStatus], tags=["webserver"])
def background_job_list():
    return [
        {
            "job_id": jobid,
            "job_category": job["category"],
            "status": job["status"],
            "running": job["is_running"],
            "progress": job.get("progress"),
            "progress_tasks": job.get("progress_tasks"),
            "error": job.get("error", None),
        }
        for jobid, job in ApiBG.jobs.items()
    ]


@router.get("/background/{jobid}", response_model=BackgroundTaskStatus, tags=["webserver"])
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": jobid,
        "job_category": job["category"],
        "status": job["status"],
        "running": job["is_running"],
        "progress": job.get("progress"),
        "progress_tasks": job.get("progress_tasks"),
        "error": job.get("error", None),
    }
