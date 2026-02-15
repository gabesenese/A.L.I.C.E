"""
Speech Facade for A.L.I.C.E
Voice input/output management
"""

from speech.stt import listen_for_voice, STTEngine
from speech.tts import speak
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SpeechFacade:
    """Facade for voice I/O systems"""

    def __init__(self) -> None:
        # STT Engine
        try:
            self.stt = STTEngine()
        except Exception as e:
            logger.warning(f"STT engine not available: {e}")
            self.stt = None

        logger.info("[SpeechFacade] Initialized speech systems")

    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input

        Args:
            timeout: Listening timeout in seconds

        Returns:
            Transcribed text or None
        """
        try:
            return listen_for_voice(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to listen for voice: {e}")
            return None

    def speak(self, text: str, wait: bool = False) -> bool:
        """
        Speak text using TTS

        Args:
            text: Text to speak
            wait: Wait for speech to finish

        Returns:
            True if speech started successfully
        """
        try:
            speak(text, wait=wait)
            return True
        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return False

    def is_stt_available(self) -> bool:
        """Check if STT is available"""
        return self.stt is not None

    def configure_stt(
        self,
        engine: str = "google",
        language: str = "en-US"
    ) -> bool:
        """
        Configure STT engine

        Args:
            engine: Engine to use (google, sphinx, etc.)
            language: Language code

        Returns:
            True if configured successfully
        """
        if not self.stt:
            return False

        try:
            self.stt.configure(engine=engine, language=language)
            return True
        except Exception as e:
            logger.error(f"Failed to configure STT: {e}")
            return False


# Singleton instance
_speech_facade: Optional[SpeechFacade] = None


def get_speech_facade() -> SpeechFacade:
    """Get or create the SpeechFacade singleton"""
    global _speech_facade
    if _speech_facade is None:
        _speech_facade = SpeechFacade()
    return _speech_facade
