from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status


async def get_current_user(request: Request) -> str:
    user_id = request.headers.get("x-user-id", "anonymous").strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity",
        )
    return user_id


def get_pipeline(request: Request):
    return request.app.state.container.pipeline
