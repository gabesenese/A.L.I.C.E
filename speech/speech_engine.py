"""
Advanced Speech Engine for A.L.I.C.E
Features:
- Speech-to-Text (STT) using Whisper or Google Speech
- Text-to-Speech (TTS) with multiple voices
- Wake word detection ("Hey Alice", "Alice")
- Real-time audio processing
- Voice activity detection
"""

import os
import logging
import threading
import queue
import time
from typing import Optional, Callable
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechConfig:
    """Configuration for speech engine"""
    def __init__(
        self,
        wake_words: list = None,
        stt_engine: str = "google",  # "google", "whisper", "vosk"
        tts_engine: str = "pyttsx3",  # "pyttsx3", "gtts", "elevenlabs"
        language: str = "en-US",
        voice_id: Optional[str] = None,
        listening_timeout: int = 5,
        phrase_timeout: int = 3
    ):
        self.wake_words = wake_words or ["alice", "hey alice"]
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.language = language
        self.voice_id = voice_id
        self.listening_timeout = listening_timeout
        self.phrase_timeout = phrase_timeout


class SpeechEngine:
    """
    Advanced speech interaction engine
    Handles voice input/output for natural conversation
    """
    
    def __init__(self, config: Optional[SpeechConfig] = None):
        self.config = config or SpeechConfig()
        self.is_listening = False
        self.wake_word_detected = False
        
        # Audio components (lazy loading)
        self._recognizer = None
        self._tts_engine = None
        self._microphone = None
        
        # Threading
        self.audio_queue = queue.Queue()
        self.listen_thread = None
        
        logger.info("✅ Speech Engine initialized")
    
    @property
    def recognizer(self):
        """Lazy load speech recognizer"""
        if self._recognizer is None:
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = 4000
                self._recognizer.dynamic_energy_threshold = True
                self._recognizer.pause_threshold = 0.8
                logger.info("🎤 Speech recognizer initialized")
            except ImportError:
                logger.error("❌ speech_recognition not installed. Run: pip install SpeechRecognition")
        return self._recognizer
    
    @property
    def microphone(self):
        """Lazy load microphone"""
        if self._microphone is None:
            try:
                import speech_recognition as sr
                self._microphone = sr.Microphone()
                logger.info("🎙Microphone initialized")
            except ImportError:
                logger.error("❌ speech_recognition not installed")
        return self._microphone
    
    @property
    def tts_engine(self):
        """Lazy load TTS engine"""
        if self._tts_engine is None:
            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                
                # Configure voice
                voices = self._tts_engine.getProperty('voices')
                
                # Try to find a good voice (prefer female voices for ALICE)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self._tts_engine.setProperty('voice', voice.id)
                        break
                
                # Set rate and volume
                self._tts_engine.setProperty('rate', 175)  # Speed
                self._tts_engine.setProperty('volume', 0.9)  # Volume
                
                logger.info("🔊 TTS engine initialized")
            except ImportError:
                logger.error("❌ pyttsx3 not installed. Run: pip install pyttsx3")
            except Exception as e:
                logger.error(f"❌ TTS initialization error: {e}")
        return self._tts_engine
    
    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            blocking: Wait for speech to finish
        """
        if not self.tts_engine:
            logger.warning("⚠TTS engine not available")
            print(f"ALICE: {text}")  # Fallback to print
            return
        
        try:
            logger.info(f"🔊 Speaking: {text}")
            
            if blocking:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                # Non-blocking speech in separate thread
                def speak_thread():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                thread = threading.Thread(target=speak_thread, daemon=True)
                thread.start()
                
        except Exception as e:
            logger.error(f"❌ Speech error: {e}")
            print(f"ALICE: {text}")  # Fallback
    
    def listen(self, timeout: Optional[int] = None, phrase_timeout: Optional[int] = None) -> Optional[str]:
        """
        Listen for speech input
        
        Args:
            timeout: Maximum time to wait for speech
            phrase_timeout: Maximum time of silence after speech
            
        Returns:
            Recognized text or None
        """
        if not self.recognizer or not self.microphone:
            logger.error("❌ Speech recognition not available")
            return None
        
        timeout = timeout or self.config.listening_timeout
        phrase_timeout = phrase_timeout or self.config.phrase_timeout
        
        try:
            with self.microphone as source:
                logger.info("🎤 Listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_timeout
                )
                
                logger.info("Processing audio...")
                
                # Recognize speech
                if self.config.stt_engine == "google":
                    text = self.recognizer.recognize_google(audio, language=self.config.language)
                elif self.config.stt_engine == "whisper":
                    text = self.recognizer.recognize_whisper(audio, language=self.config.language)
                else:
                    text = self.recognizer.recognize_google(audio)
                
                logger.info(f"✅ Recognized: {text}")
                return text
                
        except Exception as sr_error:
            # Check specific error types
            error_name = type(sr_error).__name__
            
            if "UnknownValueError" in error_name:
                logger.debug("❓ Could not understand audio")
                return None
            elif "RequestError" in error_name:
                logger.error(f"❌ Speech recognition service error: {sr_error}")
                return None
            elif "WaitTimeoutError" in error_name:
                logger.debug("⏱Listening timeout")
                return None
            else:
                logger.error(f"❌ Speech recognition error: {sr_error}")
                return None
    
    def listen_for_wake_word(self, callback: Callable[[str], None], background: bool = True):
        """
        Continuously listen for wake word
        
        Args:
            callback: Function to call when wake word detected
            background: Run in background thread
        """
        def wake_word_loop():
            logger.info(f"👂 Listening for wake words: {', '.join(self.config.wake_words)}")
            self.is_listening = True
            
            while self.is_listening:
                try:
                    text = self.listen(timeout=10, phrase_timeout=2)
                    
                    if text:
                        text_lower = text.lower()
                        
                        # Check for wake word
                        for wake_word in self.config.wake_words:
                            if wake_word.lower() in text_lower:
                                logger.info(f"[WAKE] Wake word detected: {wake_word}")
                                self.wake_word_detected = True
                                
                                # Remove wake word from text
                                command = text_lower.replace(wake_word.lower(), "").strip()
                                
                                # Call callback with command
                                callback(command if command else text)
                                break
                        
                except KeyboardInterrupt:
                    logger.info("⏹Stopping wake word detection")
                    self.is_listening = False
                    break
                except Exception as e:
                    logger.error(f"❌ Wake word detection error: {e}")
                    time.sleep(1)
        
        if background:
            self.listen_thread = threading.Thread(target=wake_word_loop, daemon=True)
            self.listen_thread.start()
            logger.info("🎙Wake word detection started in background")
        else:
            wake_word_loop()
    
    def stop_listening(self):
        """Stop wake word detection"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
        logger.info("⏹Stopped listening")
    
    def test_audio(self):
        """Test audio input/output"""
        logger.info("🧪 Testing audio system...")
        
        # Test TTS
        print("\n1. Testing Text-to-Speech...")
        self.speak("Hello! I am ALICE. Audio system test in progress.")
        
        # Test microphone
        print("\n2. Testing microphone...")
        print("   Please say something...")
        
        text = self.listen(timeout=5)
        if text:
            print(f"   ✅ Heard: {text}")
            self.speak(f"I heard you say: {text}")
        else:
            print("   ❌ Could not detect speech")
        
        print("\n✅ Audio test complete")
    
    def set_voice(self, voice_name: str):
        """Change TTS voice"""
        if self.tts_engine:
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if voice_name.lower() in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    logger.info(f"🎭 Voice changed to: {voice.name}")
                    return True
            logger.warning(f"⚠Voice '{voice_name}' not found")
        return False
    
    def list_voices(self):
        """List available TTS voices"""
        if self.tts_engine:
            voices = self.tts_engine.getProperty('voices')
            print("\n🎭 Available voices:")
            for i, voice in enumerate(voices):
                print(f"  {i+1}. {voice.name} ({voice.id})")
        else:
            logger.warning("⚠TTS engine not available")
    
    def save_audio(self, text: str, filename: str):
        """Save speech to audio file"""
        if self.tts_engine:
            try:
                self.tts_engine.save_to_file(text, filename)
                self.tts_engine.runAndWait()
                logger.info(f"Audio saved to: {filename}")
            except Exception as e:
                logger.error(f"❌ Error saving audio: {e}")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("🎤 A.L.I.C.E Speech Engine Test")
    print("=" * 80)
    
    # Create speech engine
    config = SpeechConfig(
        wake_words=["alice", "hey alice", "ok alice"],
        stt_engine="google",
        tts_engine="pyttsx3"
    )
    
    speech = SpeechEngine(config)
    
    # List available voices
    speech.list_voices()
    
    # Test audio system
    print("\n" + "=" * 80)
    speech.test_audio()
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Options:")
    print("  1. Test wake word detection")
    print("  2. Single listen test")
    print("  3. Speak test")
    print("  4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n👂 Say one of the wake words:")
            print(f"   {', '.join(config.wake_words)}")
            print("   Then say a command after the wake word")
            print("   (Press Ctrl+C to stop)\n")
            
            def handle_command(command):
                speech.speak(f"You said: {command}")
                print(f"✅ Command received: {command}")
            
            try:
                speech.listen_for_wake_word(handle_command, background=False)
            except KeyboardInterrupt:
                print("\n⏹Stopped")
        
        elif choice == "2":
            print("\n🎤 Listening... (speak now)")
            text = speech.listen()
            if text:
                print(f"✅ You said: {text}")
                speech.speak(f"You said: {text}")
            else:
                print("❌ No speech detected")
        
        elif choice == "3":
            text = input("\nEnter text to speak: ").strip()
            if text:
                speech.speak(text)
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")



