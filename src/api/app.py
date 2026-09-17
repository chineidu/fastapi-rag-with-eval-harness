"""FastAPI application factory for the RAG service."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import create_logger
from src.api.core.exceptions import (
    BaseAPIError,
    aapi_error_handler,
    ahttp_error_handler,
    arequest_validation_handler,
    aunhandled_exception_handler,
)
from src.api.core.lifespan import lifespan
from src.api.core.middleware import RequestIDMiddleware
from src.api.routes import ui
from src.api.routes.v1 import ask, health, stream
from src.app.adapter import LocalRetriever
from src.config import app_config

logger = create_logger(__name__)


def create_app(retriever: LocalRetriever) -> FastAPI:
    """Build the FastAPI app with routes, middleware, and error handlers wired.

    The caller builds the retriever and shares it via ``app.state``;
    lifespan only probes it so misconfiguration surfaces in logs.
    """
    settings = app_config
    prefix = "/" + settings.api_config.prefix.strip("/")
    app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)
    app.state.retriever = retriever

    # Register middleware (outermost last).
    app.add_middleware(RequestIDMiddleware)
    cors = settings.api_config.middleware.cors
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors.allow_origins,
        allow_credentials=cors.allow_credentials,
        allow_methods=cors.allow_methods,
        allow_headers=cors.allow_headers,
    )

    # Register handlers: specific first, catch-all Exception last.
    app.add_exception_handler(
        BaseAPIError,
        aapi_error_handler,  # type: ignore  # safe: dispatched by exc class.
    )
    app.add_exception_handler(
        StarletteHTTPException,
        ahttp_error_handler,  # type: ignore  # safe: dispatched by exc class.
    )
    app.add_exception_handler(
        RequestValidationError,
        arequest_validation_handler,  # type: ignore  # safe: dispatched by exc class.
    )
    app.add_exception_handler(Exception, aunhandled_exception_handler)

    app.include_router(health.router, prefix=prefix)
    app.include_router(ask.router, prefix=prefix)
    app.include_router(stream.router, prefix=prefix)
    app.include_router(ui.router)
    logger.debug("Routes registered under prefix %s", prefix)
    return app
