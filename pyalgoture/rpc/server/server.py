from ipaddress import ip_address
from typing import Any

import orjson
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from ...utils.logger import get_logger
from ..rpc import RPC, RPCException, RPCHandler
from ..rpc_types import RPCSendMsg
from .uvicorn_threaded import UvicornServer
from .webserver_bgwork import ApiBG


class OperationalException(Exception):
    """
    Requires manual intervention and will stop the bot.
    Most of the time, this is caused by an invalid Configuration.
    """


def running_in_docker() -> bool:
    """
    Check if we are running in a docker container
    """
    return False


logger = get_logger()


class CustomJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        Use rapidjson for responses
        Handles NaN and Inf / -Inf in a javascript way by default.
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer(RPCHandler):
    __instance: "ApiServer | None" = None
    __initialized = False

    _rpc: RPC | None = None
    _has_rpc: bool = False
    _config: dict[str, Any] = {}

    def __new__(cls, *args, **kwargs) -> "ApiServer":
        """
        This class is a singleton.
        We'll only have one instance of it around.
        """
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: dict[str, Any], standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server: UvicornServer | None = None

        ApiServer.__initialized = True

        api_config = self._config["api_server"]

        self.app = FastAPI(
            title="Trade Statistics API",
            docs_url="/docs" if api_config.get("enable_openapi", False) else None,
            redoc_url=None,
            default_response_class=CustomJSONResponse,
        )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC) -> None:
        """
        Attach rpc handler
        """
        if not ApiServer._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            # This should not happen assuming we didn't mess up.
            raise OperationalException("RPC Handler already attached.")

    def cleanup(self) -> None:
        """Cleanup pending module resources"""
        ApiServer._has_rpc = False
        if hasattr(ApiBG, "exchanges"):
            ApiBG.exchanges = {}
        if hasattr(ApiBG, "jobs"):
            ApiBG.jobs = {}
        if self._server and not self._standalone:
            logger.info("Stopping API Server")
            # self._server.force_exit, self._server.should_exit = True, True
            self._server.cleanup()

    @classmethod
    def shutdown(cls) -> None:
        cls.__initialized = False
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        Publish the message to the message stream
        """
        pass

    def handle_rpc_exception(self, request: Request, exc: RPCException) -> JSONResponse:
        logger.error(f"API({request.url.path}) Error calling: {exc}")
        return JSONResponse(
            status_code=502,
            content={"error": f"Error querying {request.url.path}: {exc.message}"},
        )

    def configure_app(self, app: FastAPI, config: dict[str, Any]) -> None:
        from .api_auth import http_basic_or_jwt_token, router_login
        from .api_background_tasks import router as api_bg_tasks
        from .api_reconciliation import router as api_reconciliation
        from .api_v1 import router as api_v1
        from .api_v1 import router_public as api_v1_public

        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(router_login, prefix="/api/v1", tags=["auth"])
        app.include_router(
            api_v1,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token)],
        )
        app.include_router(
            api_bg_tasks,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token)],
        )
        app.include_router(
            api_reconciliation,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token)],
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config["api_server"].get("CORS_origins", []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)
        app.add_event_handler(event_type="startup", func=self._api_startup_event)
        app.add_event_handler(event_type="shutdown", func=self._api_shutdown_event)

    async def _api_startup_event(self) -> None:
        """
        Creates the MessageStream class on startup
        so it has access to the same event loop
        as uvicorn
        """
        pass

    async def _api_shutdown_event(self) -> None:
        """
        Removes the MessageStream class on shutdown
        """
        pass

    def start_api(self) -> None:
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config["api_server"]["listen_ip_address"]
        rest_port = self._config["api_server"]["listen_port"]

        logger.info(f"Starting HTTP Server at {rest_ip}:{rest_port}")
        if not ip_address(rest_ip).is_loopback and not running_in_docker():
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning(
                "SECURITY WARNING - This is insecure please set to your loopback, e.g 127.0.0.1 in config.json"
            )

        if not self._config["api_server"].get("password"):
            logger.warning(
                "SECURITY WARNING - No password for local REST Server defined. Please make sure that this is intentional!"
            )

        if self._config["api_server"].get("jwt_secret_key", "super-secret") in ("super-secret, somethingrandom"):
            logger.warning(
                "SECURITY WARNING - `jwt_secret_key` seems to be default. Others may be able to log into your bot."
            )

        logger.info("Starting Local Rest Server.")
        verbosity = self._config["api_server"].get("verbosity", "error")

        uvconfig = uvicorn.Config(
            self.app,
            port=rest_port,
            host=rest_ip,
            use_colors=False,
            log_config=None,
            access_log=True if verbosity != "error" else False,
            ws_ping_interval=None,  # We do this explicitly ourselves
        )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")
