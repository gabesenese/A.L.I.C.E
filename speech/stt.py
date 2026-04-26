from __future__ import annotations


class WhisperSTT:
    async def record_until_silence(self) -> bytes:
        return b""

    async def transcribe(self, audio: bytes) -> str:
        return ""
