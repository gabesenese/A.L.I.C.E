from __future__ import annotations

from fastapi import FastAPI

from app.api.middleware import register_middleware
from app.api.routes import router
from app.config import get_settings
from app.container import AppContainer, build_container
from app.lifespan import app_lifespan
from app.logging_config import configure_logging
from fastapi import Request


def create_container() -> AppContainer:
    settings = get_settings()
    configure_logging(settings.log_level)
    return build_container(settings)


def create_app() -> FastAPI:
    container = create_container()
    app = FastAPI(title="A.L.I.C.E.", version="2.0.0", lifespan=app_lifespan)
    app.state.container = container
    register_middleware(app)
    app.include_router(router)
    return app


def get_pipeline(request: Request):
    return request.app.state.container.pipeline
