from __future__ import annotations

import httpx

from speech.stt import WhisperSTT
from speech.tts import KokoroTTS


class WakeWordDetector:
    def __init__(self, keyword: str = "alice") -> None:
        self.keyword = keyword

    async def listen(self) -> bool:
        return False


class VoiceGateway:
    def __init__(self, api_base: str = "http://localhost:8000") -> None:
        self.wake = WakeWordDetector(keyword="alice")
        self.stt = WhisperSTT()
        self.tts = KokoroTTS()
        self.client = httpx.AsyncClient(base_url=api_base)
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            if await self.wake.listen():
                audio = await self.stt.record_until_silence()
                text = await self.stt.transcribe(audio)
                response = await self.client.post(
                    "/chat",
                    json={"message": text, "user_id": "voice_user"},
                )
                data = response.json()
                await self.tts.speak(str(data.get("response", "")))

    async def stop(self) -> None:
        self._running = False
        await self.client.aclose()
