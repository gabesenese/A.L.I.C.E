"""
Speech Facade for A.L.I.C.E
Voice input/output management
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from speech.speech_engine import SpeechEngine
    _speech_available = True
except ImportError:
    _speech_available = False


class SpeechFacade:
    """Facade for voice I/O systems"""

    def __init__(self) -> None:
        try:
            self.engine = SpeechEngine() if _speech_available else None
        except Exception as e:
            logger.warning(f"Speech engine not available: {e}")
            self.engine = None

        logger.info(f"[SpeechFacade] Initialized (available={self.engine is not None})")

    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for voice input

        Args:
            timeout: Listening timeout in seconds

        Returns:
            Transcribed text or None
        """
        if not self.engine:
            return None

        try:
            return self.engine.listen(timeout=timeout)
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
        if not self.engine:
            return False

        try:
            self.engine.speak(text)
            return True
        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return False

    def is_available(self) -> bool:
        """Check if speech engine is available"""
        return self.engine is not None


# Singleton instance
_speech_facade: Optional[SpeechFacade] = None


def get_speech_facade() -> SpeechFacade:
    """Get or create the SpeechFacade singleton"""
    global _speech_facade
    if _speech_facade is None:
        _speech_facade = SpeechFacade()
    return _speech_facade
