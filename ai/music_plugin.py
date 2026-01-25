"""
A.L.I.C.E Music Control Plugin

This plugin provides comprehensive music control capabilities including:
- Spotify integration (with API authentication)
- Local music file playback
- Interactive local music selection (list .mp3 files, prompt user to choose, play selected file)
- Playlist management
- Volume control
- Search and discovery
- Natural language music commands

Supports commands like:
- "Play some jazz music"
- "Skip to the next song"
- "Set volume to 50%"
- "Play my workout playlist"
- "Search for songs by The Beatles"
- "List my music" (shows all .mp3 files and lets you choose which to play)
"""

import os
import json
import time
import random
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Spotify API imports (optional - will gracefully degrade if not available)
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    print("Spotify integration not available. Install spotipy: pip install spotipy")

# Local music playback imports (optional - will gracefully degrade if not available)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Local music playback not available. Install pygame: pip install pygame")

# Web browser automation for YouTube Music
try:
    import webbrowser
    import urllib.parse
    WEB_CONTROL_AVAILABLE = True
except ImportError:
    WEB_CONTROL_AVAILABLE = False

# Desktop control imports (for controlling Spotify desktop app without API)
try:
    import subprocess
    import win32gui
    import win32con
    import win32api
    DESKTOP_CONTROL_AVAILABLE = True
except ImportError:
    DESKTOP_CONTROL_AVAILABLE = False
    print("Desktop control not available. Install pywin32: pip install pywin32")

# Audio file metadata imports
try:
    import mutagen
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.flac import FLAC
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    print("Music metadata not available. Install mutagen: pip install mutagen")

class YouTubeMusicController:
    """Controls YouTube Music via web browser (no API required)"""
    
    def __init__(self):
        self.is_available = WEB_CONTROL_AVAILABLE
        self.youtube_music_url = "https://music.youtube.com"
        self.youtube_url = "https://www.youtube.com"
    
    def search_and_play(self, query: str, use_music_app: bool = True) -> bool:
        """Search and play music on YouTube or YouTube Music"""
        if not self.is_available:
            return False
        
        try:
            # URL encode the search query
            encoded_query = urllib.parse.quote_plus(query)
            
            if use_music_app:
                # Use YouTube Music for better music experience
                search_url = f"{self.youtube_music_url}/search?q={encoded_query}"
            else:
                # Use regular YouTube
                search_url = f"{self.youtube_url}/results?search_query={encoded_query}"
            
            # Open in default browser
            webbrowser.open(search_url)
            return True
            
        except Exception as e:
            print(f"Error opening YouTube Music: {e}")
            return False
    
    def open_youtube_music(self) -> bool:
        """Open YouTube Music homepage"""
        try:
            webbrowser.open(self.youtube_music_url)
            return True
        except:
            return False
    
    def play_specific_song(self, song: str, artist: str = "") -> bool:
        """Play a specific song by artist on YouTube Music"""
        if artist:
            query = f"{song} {artist}"
        else:
            query = song
        
        return self.search_and_play(query, use_music_app=True)
    
    def get_playlist_url(self, playlist_name: str) -> str:
        """Generate YouTube Music playlist search URL"""
        encoded_query = urllib.parse.quote_plus(f"playlist {playlist_name}")
        return f"{self.youtube_music_url}/search?q={encoded_query}"

class DesktopSpotifyController:
    """Controls Spotify desktop app using system media keys (no API required)"""
    
    def __init__(self):
        self.is_available = DESKTOP_CONTROL_AVAILABLE
    
    def is_spotify_running(self) -> bool:
        """Check if Spotify desktop app is running"""
        try:
            def enum_window_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if 'spotify' in window_title.lower():
                        windows.append((hwnd, window_title))
                return True
            
            windows = []
            win32gui.EnumWindows(enum_window_callback, windows)
            return len(windows) > 0
        except:
            return False
    
    def send_media_key(self, key_code):
        """Send media key to control any media player"""
        if not self.is_available:
            return False
        
        try:
            # Send key down and key up
            win32api.keybd_event(key_code, 0, 0, 0)
            win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
            return True
        except:
            return False
    
    def play_pause(self) -> bool:
        """Toggle play/pause using media key"""
        return self.send_media_key(win32con.VK_MEDIA_PLAY_PAUSE)
    
    def next_track(self) -> bool:
        """Skip to next track using media key"""
        return self.send_media_key(win32con.VK_MEDIA_NEXT_TRACK)
    
    def previous_track(self) -> bool:
        """Go to previous track using media key"""
        return self.send_media_key(win32con.VK_MEDIA_PREV_TRACK)
    
    def stop(self) -> bool:
        """Stop playback using media key"""
        return self.send_media_key(win32con.VK_MEDIA_STOP)
    
    def search_and_play(self, query: str) -> bool:
        """Search for music in Spotify desktop app"""
        if not self.is_spotify_running():
            return False
        
        try:
            # Find Spotify window
            spotify_hwnd = None
            def find_spotify_window(hwnd, param):
                nonlocal spotify_hwnd
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if 'spotify' in window_title.lower():
                        spotify_hwnd = hwnd
                        return False  # Stop enumeration
                return True
            
            win32gui.EnumWindows(find_spotify_window, None)
            
            if spotify_hwnd:
                # Bring Spotify to foreground
                win32gui.SetForegroundWindow(spotify_hwnd)
                
                # Send Ctrl+L to focus search box
                win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
                win32api.keybd_event(ord('L'), 0, 0, 0)
                win32api.keybd_event(ord('L'), 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                time.sleep(0.5)
                
                # Type the search query
                for char in query:
                    if char == ' ':
                        win32api.keybd_event(win32con.VK_SPACE, 0, 0, 0)
                        win32api.keybd_event(win32con.VK_SPACE, 0, win32con.KEYEVENTF_KEYUP, 0)
                    else:
                        vk_code = ord(char.upper())
                        win32api.keybd_event(vk_code, 0, 0, 0)
                        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                    time.sleep(0.05)
                
                time.sleep(0.5)
                
                # Press Enter to search and play first result
                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                time.sleep(0.5)
                
                # Press Enter again to play first result
                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                
                return True
        except Exception as e:
            print(f"Error controlling desktop Spotify: {e}")
        
        return False

# Import the proper plugin interface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.plugin_system import PluginInterface

@dataclass
class Song:
    """Represents a music track"""
    title: str
    artist: str
    album: str = ""
    duration: int = 0  # in seconds
    uri: str = ""  # Spotify URI or file path
    source: str = ""  # "spotify", "local", "youtube", etc.
    genre: str = ""
    year: int = 0
    track_number: int = 0
    
    def __str__(self) -> str:
        return f"{self.title} by {self.artist}"

@dataclass
class Playlist:
    """Represents a music playlist"""
    name: str
    songs: List[Song]
    source: str = ""  # "spotify", "local", etc.
    uri: str = ""
    description: str = ""
    
    def add_song(self, song: Song) -> None:
        self.songs.append(song)
    
    def remove_song(self, index: int) -> bool:
        if 0 <= index < len(self.songs):
            self.songs.pop(index)
            return True
        return False
    
    def __str__(self) -> str:
        return f"Playlist '{self.name}' ({len(self.songs)} songs)"

class SpotifyManager:
    """Handles Spotify API integration"""
    
    def __init__(self):
        self.spotify = None
        self.is_authenticated = False
        self.credentials_path = "cred/spotify_credentials.json"
        self.setup_spotify()
    
    def setup_spotify(self) -> bool:
        """Initialize Spotify API with credentials"""
        if not SPOTIFY_AVAILABLE:
            return False
            
        try:
            # Check for credentials file
            if not os.path.exists(self.credentials_path):
                print(f"Spotify credentials not found at {self.credentials_path}")
                return False
            
            with open(self.credentials_path, 'r') as f:
                creds = json.load(f)
            
            # Set up Spotify OAuth
            client_id = creds.get('client_id')
            client_secret = creds.get('client_secret')
            redirect_uri = creds.get('redirect_uri', 'http://localhost:8888/callback')
            
            if not client_id or not client_secret:
                print("Invalid Spotify credentials")
                return False
            
            # Create Spotify client with OAuth
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-read-playback-state,user-modify-playback-state,user-read-currently-playing,playlist-read-private,playlist-read-collaborative,user-library-read",
                cache_path="cred/spotify_token.cache"
            )
            
            self.spotify = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test authentication
            try:
                user = self.spotify.current_user()
                self.is_authenticated = True
                print(f"Spotify connected for user: {user.get('display_name', 'Unknown')}")
                return True
            except Exception as e:
                print(f"Spotify authentication failed: {e}")
                return False
                
        except Exception as e:
            print(f"Spotify setup error: {e}")
            return False
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Song]:
        """Search for tracks on Spotify"""
        if not self.is_authenticated:
            return []
        
        try:
            results = self.spotify.search(q=query, type='track', limit=limit)
            songs = []
            
            for track in results['tracks']['items']:
                song = Song(
                    title=track['name'],
                    artist=', '.join([artist['name'] for artist in track['artists']]),
                    album=track['album']['name'],
                    duration=track['duration_ms'] // 1000,
                    uri=track['uri'],
                    source="spotify",
                    year=int(track['album']['release_date'][:4]) if track['album'].get('release_date') else 0
                )
                songs.append(song)
            
            return songs
        except Exception as e:
            print(f"Spotify search error: {e}")
            return []
    
    def get_user_playlists(self) -> List[Playlist]:
        """Get user's Spotify playlists"""
        if not self.is_authenticated:
            return []
        
        try:
            playlists_result = self.spotify.current_user_playlists()
            playlists = []
            
            for playlist_data in playlists_result['items']:
                playlist = Playlist(
                    name=playlist_data['name'],
                    songs=[],  # We'll load songs on demand
                    source="spotify",
                    uri=playlist_data['uri'],
                    description=playlist_data.get('description', '')
                )
                playlists.append(playlist)
            
            return playlists
        except Exception as e:
            print(f"Error getting playlists: {e}")
            return []
    
    def get_playlist_tracks(self, playlist_uri: str) -> List[Song]:
        """Get tracks from a specific playlist"""
        if not self.is_authenticated:
            return []
        
        try:
            playlist_id = playlist_uri.split(':')[-1]
            results = self.spotify.playlist_tracks(playlist_id)
            songs = []
            
            for item in results['items']:
                if item['track'] and item['track']['type'] == 'track':
                    track = item['track']
                    song = Song(
                        title=track['name'],
                        artist=', '.join([artist['name'] for artist in track['artists']]),
                        album=track['album']['name'],
                        duration=track['duration_ms'] // 1000,
                        uri=track['uri'],
                        source="spotify"
                    )
                    songs.append(song)
            
            return songs
        except Exception as e:
            print(f"Error getting playlist tracks: {e}")
            return []
    
    def play_track(self, track_uri: str) -> bool:
        """Play a specific track on Spotify"""
        if not self.is_authenticated:
            return False
        
        try:
            # Try to play on active device
            self.spotify.start_playback(uris=[track_uri])
            return True
        except Exception as e:
            print(f"Error playing track: {e}")
            return False
    
    def pause_playback(self) -> bool:
        """Pause Spotify playback"""
        if not self.is_authenticated:
            return False
        
        try:
            self.spotify.pause_playback()
            return True
        except Exception as e:
            print(f"Error pausing playback: {e}")
            return False
    
    def resume_playback(self) -> bool:
        """Resume Spotify playback"""
        if not self.is_authenticated:
            return False
        
        try:
            self.spotify.start_playback()
            return True
        except Exception as e:
            print(f"Error resuming playback: {e}")
            return False
    
    def next_track(self) -> bool:
        """Skip to next track"""
        if not self.is_authenticated:
            return False
        
        try:
            self.spotify.next_track()
            return True
        except Exception as e:
            print(f"Error skipping track: {e}")
            return False
    
    def previous_track(self) -> bool:
        """Go to previous track"""
        if not self.is_authenticated:
            return False
        
        try:
            self.spotify.previous_track()
            return True
        except Exception as e:
            print(f"Error going to previous track: {e}")
            return False
    
    def set_volume(self, volume: int) -> bool:
        """Set Spotify volume (0-100)"""
        if not self.is_authenticated:
            return False
        
        try:
            volume = max(0, min(100, volume))  # Clamp between 0-100
            self.spotify.volume(volume)
            return True
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False
    
    def get_current_track(self) -> Optional[Song]:
        """Get currently playing track"""
        if not self.is_authenticated:
            return None
        
        try:
            current = self.spotify.current_playback()
            if current and current['item']:
                track = current['item']
                return Song(
                    title=track['name'],
                    artist=', '.join([artist['name'] for artist in track['artists']]),
                    album=track['album']['name'],
                    duration=track['duration_ms'] // 1000,
                    uri=track['uri'],
                    source="spotify"
                )
        except Exception as e:
            print(f"Error getting current track: {e}")
        
        return None

class LocalMusicManager:
    """Handles local music file playback"""
    
    def __init__(self):
        self.is_initialized = False
        self.current_song = None
        self.is_playing = False
        self.is_paused = False
        self.volume = 70
        self.position = 0
        
        # Common music directories
        self.music_directories = [
            os.path.expanduser("~/Music"),
            os.path.expanduser("~/Documents/Music"),
            "C:/Users/*/Music",
            "./music"
        ]
        
        self.supported_formats = {'.mp3', '.mp4', '.m4a', '.flac', '.wav', '.ogg'}
        self.music_library = []
        
        if PYGAME_AVAILABLE:
            self.init_pygame()
            self.scan_music_library()
    
    def init_pygame(self) -> bool:
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize audio playback: {e}")
            return False
    
    def scan_music_library(self) -> None:
        """Scan local directories for music files"""
        self.music_library = []
        
        for directory in self.music_directories:
            directory = os.path.expanduser(directory)
            if os.path.exists(directory):
                self._scan_directory(directory)
        
        print(f"Found {len(self.music_library)} local music files")
    
    def _scan_directory(self, directory: str) -> None:
        """Recursively scan directory for music files"""
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_formats):
                        file_path = os.path.join(root, file)
                        song = self._create_song_from_file(file_path)
                        if song:
                            self.music_library.append(song)
        except PermissionError:
            # Skip directories we don't have permission to read
            pass
    
    def _create_song_from_file(self, file_path: str) -> Optional[Song]:
        """Create Song object from music file"""
        try:
            # Basic info from filename
            filename = os.path.basename(file_path)
            title = os.path.splitext(filename)[0]
            
            # Try to extract metadata if mutagen is available
            if METADATA_AVAILABLE:
                try:
                    if file_path.lower().endswith('.mp3'):
                        audio_file = MP3(file_path)
                        title = str(audio_file.get('TIT2', [title])[0]) if audio_file.get('TIT2') else title
                        artist = str(audio_file.get('TPE1', ['Unknown Artist'])[0]) if audio_file.get('TPE1') else 'Unknown Artist'
                        album = str(audio_file.get('TALB', ['Unknown Album'])[0]) if audio_file.get('TALB') else 'Unknown Album'
                        duration = int(audio_file.info.length) if hasattr(audio_file, 'info') else 0
                    
                    elif file_path.lower().endswith(('.m4a', '.mp4')):
                        audio_file = MP4(file_path)
                        title = audio_file.get('\xa9nam', [title])[0] if audio_file.get('\xa9nam') else title
                        artist = audio_file.get('\xa9ART', ['Unknown Artist'])[0] if audio_file.get('\xa9ART') else 'Unknown Artist'
                        album = audio_file.get('\xa9alb', ['Unknown Album'])[0] if audio_file.get('\xa9alb') else 'Unknown Album'
                        duration = int(audio_file.info.length) if hasattr(audio_file, 'info') else 0
                    
                    elif file_path.lower().endswith('.flac'):
                        audio_file = FLAC(file_path)
                        title = audio_file.get('TITLE', [title])[0] if audio_file.get('TITLE') else title
                        artist = audio_file.get('ARTIST', ['Unknown Artist'])[0] if audio_file.get('ARTIST') else 'Unknown Artist'
                        album = audio_file.get('ALBUM', ['Unknown Album'])[0] if audio_file.get('ALBUM') else 'Unknown Album'
                        duration = int(audio_file.info.length) if hasattr(audio_file, 'info') else 0
                    
                    else:
                        # Fallback for other formats
                        artist = 'Unknown Artist'
                        album = 'Unknown Album'
                        duration = 0
                        
                except Exception:
                    # Fallback if metadata extraction fails
                    artist = 'Unknown Artist'
                    album = 'Unknown Album'
                    duration = 0
            else:
                artist = 'Unknown Artist'
                album = 'Unknown Album'
                duration = 0
            
            return Song(
                title=title,
                artist=artist,
                album=album,
                duration=duration,
                uri=file_path,
                source="local"
            )
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def search_local_music(self, query: str) -> List[Song]:
        """Search local music library"""
        query_lower = query.lower()
        results = []
        
        for song in self.music_library:
            # Search in title, artist, and album
            if (query_lower in song.title.lower() or 
                query_lower in song.artist.lower() or 
                query_lower in song.album.lower()):
                results.append(song)
        
        return results
    
    def play_file(self, file_path: str) -> bool:
        """Play a local music file"""
        if not self.is_initialized:
            return False
        
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(self.volume / 100)
            pygame.mixer.music.play()
            
            self.current_song = next((song for song in self.music_library if song.uri == file_path), None)
            self.is_playing = True
            self.is_paused = False
            return True
            
        except Exception as e:
            print(f"Error playing file {file_path}: {e}")
            return False
    
    def pause(self) -> bool:
        """Pause local music playback"""
        if not self.is_initialized or not self.is_playing:
            return False
        
        try:
            pygame.mixer.music.pause()
            self.is_paused = True
            return True
        except Exception as e:
            print(f"Error pausing playback: {e}")
            return False
    
    def resume(self) -> bool:
        """Resume local music playback"""
        if not self.is_initialized or not self.is_paused:
            return False
        
        try:
            pygame.mixer.music.unpause()
            self.is_paused = False
            return True
        except Exception as e:
            print(f"Error resuming playback: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop local music playback"""
        if not self.is_initialized:
            return False
        
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
            self.current_song = None
            return True
        except Exception as e:
            print(f"Error stopping playback: {e}")
            return False
    
    def set_volume(self, volume: int) -> bool:
        """Set local playback volume (0-100)"""
        if not self.is_initialized:
            return False
        
        try:
            self.volume = max(0, min(100, volume))
            pygame.mixer.music.set_volume(self.volume / 100)
            return True
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False
    
    def get_current_song(self) -> Optional[Song]:
        """Get currently playing local song"""
        if self.is_playing and self.current_song:
            return self.current_song
        return None

class MusicPlugin(PluginInterface):
    def handle_local_music_selection(self, selection: str) -> str:
        """Play the .mp3 file chosen by the user (by number)"""
        if not self.pending_local_selection:
            return "No local music selection is pending. Say 'list my music' to see available files."

        try:
            index = int(selection.strip()) - 1
            if index < 0 or index >= len(self.pending_local_selection):
                return f"Invalid selection. Please choose a number between 1 and {len(self.pending_local_selection)}."
            song = self.pending_local_selection[index]
            success = self.local_manager.play_file(song.uri)
            self.pending_local_selection = None
            if success:
                return f"▶️ Playing '{song.title}' by {song.artist} from local files."
            else:
                return f"❌ Failed to play '{song.title}'."
        except ValueError:
            return "Please reply with a valid number."

    def prompt_local_music_selection(self) -> str:
        """List all .mp3 files and prompt user to choose one to play"""
        if not self.local_manager.is_initialized:
            return "Local music playback is not available. Please install pygame and add music files to your Music folder."

        # List all .mp3 files
        mp3_files = [song for song in self.local_manager.music_library if song.uri.lower().endswith('.mp3')]
        if not mp3_files:
            return "No .mp3 files found in your music library. Add .mp3 files to ~/Music or ./music."

        # Store pending selection for next step
        self.pending_local_selection = mp3_files

        # Build prompt
        response = "🎵 Found the following .mp3 files:\n"
        for i, song in enumerate(mp3_files, 1):
            response += f"{i}. {song.title} by {song.artist} ({os.path.basename(song.uri)})\n"
        response += "\nPlease reply with the number of the song you want to play."
        return response
        # Add attribute for pending local selection
        pending_local_selection = None
    """Main music control plugin for A.L.I.C.E"""
    
    def __init__(self):
        super().__init__()
        self.name = "Music Control"
        self.version = "1.0.0"
        self.description = "Controls music via YouTube Music, Spotify desktop, local files, and media keys"
        self.enabled = True
        self.capabilities = ['music_playback', 'youtube_music', 'spotify_desktop_control', 'local_music', 'media_keys']
        
        self.youtube_controller = YouTubeMusicController()  # YouTube Music (no API)
        self.spotify_manager = SpotifyManager()  # Web API (when available)
        self.desktop_spotify = DesktopSpotifyController()  # Desktop control (no API)
        self.local_manager = LocalMusicManager()
        self.current_source = "auto"  # "youtube", "spotify", "local", or "auto"
        
        # Music-related intents this plugin can handle
        self.supported_intents = {
            'MUSIC_PLAY', 'MUSIC_PAUSE', 'MUSIC_RESUME', 'MUSIC_STOP',
            'MUSIC_NEXT', 'MUSIC_PREVIOUS', 'MUSIC_VOLUME',
            'MUSIC_SEARCH', 'MUSIC_PLAYLIST', 'MUSIC_STATUS',
            'PLAY', 'PAUSE', 'NEXT', 'PREVIOUS', 'VOLUME'
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            # Test initialization of managers
            return True
        except Exception as e:
            logger.error(f"Music plugin initialization failed: {e}")
            return False
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        # Check if intent is music-related
        if intent.upper() in self.supported_intents:
            return True
        
        # Check for music-related entities
        music_entities = entities.get('music_entities', {})
        if music_entities.get('action') or music_entities.get('song') or music_entities.get('artist'):
            return True
        
        return False
    
    def execute(self, intent: str, query: str, entities: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Execute music plugin functionality"""
        try:
            response = self.handle_request(intent, entities, query)
            return {
                'success': True,
                'response': response,
                'data': {'intent': intent, 'entities': entities}
            }
        except Exception as e:
            return {
                'success': False,
                'response': f"Music plugin error: {str(e)}",
                'data': {'error': str(e)}
            }
    
    def handle_request(self, intent: str, entities: Dict[str, Any], user_input: str) -> str:
        # Check for pending local music selection
        if self.pending_local_selection:
            # If user input is a number, handle selection
            if user_input.strip().isdigit():
                return self.handle_local_music_selection(user_input)
            # If user says cancel or exit
            if user_input.strip().lower() in {"cancel", "exit", "stop"}:
                self.pending_local_selection = None
                return "Local music selection cancelled."
            # Otherwise, prompt again
                return "Please reply with the number of the song you want to play, or say 'cancel' to exit."

        # If user asks to list local music
        if any(kw in user_input.lower() for kw in ["list my music", "show my music", "list mp3", "show mp3", "choose music", "choose song"]):
            return self.prompt_local_music_selection()
        """Handle music-related requests"""
        try:
            # Extract music entities
            music_entities = entities.get('music_entities', {})
            action = music_entities.get('action', '').lower()
            song = music_entities.get('song', '')
            artist = music_entities.get('artist', '')
            album = music_entities.get('album', '')
            playlist = music_entities.get('playlist', '')
            volume = music_entities.get('volume')
            
            # Handle different music actions
            if intent in ['MUSIC_PLAY', 'PLAY'] or action == 'play':
                return self._handle_play_request(song, artist, album, playlist, user_input)
            
            elif intent in ['MUSIC_PAUSE', 'PAUSE'] or action == 'pause':
                return self._handle_pause_request()
            
            elif intent in ['MUSIC_RESUME'] or action == 'resume':
                return self._handle_resume_request()
            
            elif intent in ['MUSIC_STOP'] or action == 'stop':
                return self._handle_stop_request()
            
            elif intent in ['MUSIC_NEXT', 'NEXT'] or action in ['next', 'skip']:
                return self._handle_next_request()
            
            elif intent in ['MUSIC_PREVIOUS', 'PREVIOUS'] or action == 'previous':
                return self._handle_previous_request()
            
            elif intent in ['MUSIC_VOLUME', 'VOLUME'] or action == 'volume':
                return self._handle_volume_request(volume, user_input)
            
            elif intent == 'MUSIC_SEARCH' or action == 'search':
                return self._handle_search_request(song, artist, album, user_input)
            
            elif intent == 'MUSIC_PLAYLIST' or action == 'playlist':
                return self._handle_playlist_request(playlist, user_input)
            
            elif intent == 'MUSIC_STATUS' or action == 'status':
                return self._handle_status_request()
            
            else:
                return "I can help you control music playback. Try commands like 'play some music', 'pause', 'next song', or 'set volume to 50%'."
        
        except Exception as e:
            return f"Sorry, I encountered an error with music control: {str(e)}"
    
    def _handle_play_request(self, song: str, artist: str, album: str, playlist: str, user_input: str) -> str:
        """Handle play requests"""
        # Set source preference based on user input
        if "youtube" in user_input.lower():
            self.current_source = "youtube"
        elif "spotify" in user_input.lower():
            self.current_source = "spotify"
        elif "local" in user_input.lower():
            self.current_source = "local"
        else:
            self.current_source = "auto"
        
        # If specific song/artist requested
        if song or artist:
            query = f"{song} {artist}".strip()
            return self._play_by_search(query)
        
        # If playlist requested
        if playlist:
            return self._play_playlist(playlist)
        
        # If album requested
        if album:
            return self._play_by_search(f"album {album}")
        
        # General play request - extract search terms from user input
        search_terms = self._extract_search_terms(user_input)
        if search_terms:
            return self._play_by_search(search_terms)
        
        # Fallback - resume playbook or suggest YouTube Music
        if self.youtube_controller.is_available:
            if self.youtube_controller.open_youtube_music():
                return "🎵 Opened YouTube Music - what would you like to listen to?"
        
        # Try resuming existing playback
        if self.spotify_manager.is_authenticated:
            if self.spotify_manager.resume_playback():
                return "▶️ Resumed Spotify playback"
        
        if self.local_manager.is_initialized:
            if self.local_manager.resume():
                return "▶️ Resumed local music playback"
        
        return "🎵 Please specify what you'd like to play (e.g., 'play some jazz' or 'play The Beatles')"
    
    def _handle_pause_request(self) -> str:
        """Handle pause requests"""
        success_spotify = False
        success_local = False
        success_desktop = False
        
        # Try desktop Spotify control first
        if self.desktop_spotify.is_available:
            success_desktop = self.desktop_spotify.play_pause()
        
        # Try Spotify Web API
        if self.spotify_manager.is_authenticated:
            success_spotify = self.spotify_manager.pause_playback()
        
        # Try local playback
        if self.local_manager.is_initialized:
            success_local = self.local_manager.pause()
        
        if success_desktop or success_spotify or success_local:
            return "⏸️ Paused music playback"
        else:
            return "❌ No active playback to pause"
    
    def _handle_resume_request(self) -> str:
        """Handle resume requests"""
        success_spotify = False
        success_local = False
        
        if self.spotify_manager.is_authenticated:
            success_spotify = self.spotify_manager.resume_playback()
        
        if self.local_manager.is_initialized:
            success_local = self.local_manager.resume()
        
        if success_spotify or success_local:
            return "▶️ Resumed music playback"
        else:
            return "❌ No paused playback to resume"
    
    def _handle_stop_request(self) -> str:
        """Handle stop requests"""
        success_local = False
        
        # Note: Spotify doesn't have a stop function, only pause
        if self.spotify_manager.is_authenticated:
            self.spotify_manager.pause_playback()
        
        if self.local_manager.is_initialized:
            success_local = self.local_manager.stop()
        
        return "⏹️ Stopped music playback"
    
    def _handle_next_request(self) -> str:
        """Handle next track requests"""
        success_desktop = False
        success_spotify = False
        
        # Try desktop control first
        if self.desktop_spotify.is_available:
            success_desktop = self.desktop_spotify.next_track()
        
        # Try Spotify Web API
        if self.spotify_manager.is_authenticated:
            success_spotify = self.spotify_manager.next_track()
        
        if success_desktop or success_spotify:
            return "⏭️ Skipped to next track"
        
        return "❌ Next track control available with Spotify desktop app or Web API"
    
    def _handle_previous_request(self) -> str:
        """Handle previous track requests"""
        success_desktop = False
        success_spotify = False
        
        # Try desktop control first
        if self.desktop_spotify.is_available:
            success_desktop = self.desktop_spotify.previous_track()
        
        # Try Spotify Web API
        if self.spotify_manager.is_authenticated:
            success_spotify = self.spotify_manager.previous_track()
        
        if success_desktop or success_spotify:
            return "⏮️ Went to previous track"
        
        return "❌ Previous track control available with Spotify desktop app or Web API"
    
    def _handle_volume_request(self, volume: Optional[int], user_input: str) -> str:
        """Handle volume control requests"""
        # Try to extract volume from user input if not provided
        if volume is None:
            volume = self._extract_volume_from_input(user_input)
        
        if volume is None:
            return "🔊 Please specify a volume level (0-100), e.g., 'set volume to 50%'"
        
        success_spotify = False
        success_local = False
        
        if self.spotify_manager.is_authenticated:
            success_spotify = self.spotify_manager.set_volume(volume)
        
        if self.local_manager.is_initialized:
            success_local = self.local_manager.set_volume(volume)
        
        if success_spotify or success_local:
            return f"🔊 Set volume to {volume}%"
        else:
            return "❌ Unable to set volume"
    
    def _handle_search_request(self, song: str, artist: str, album: str, user_input: str) -> str:
        """Handle music search requests"""
        query = f"{song} {artist} {album}".strip()
        if not query:
            query = self._extract_search_terms(user_input)
        
        if not query:
            return "🔍 Please specify what to search for"
        
        results = []
        
        # Search Spotify
        if self.spotify_manager.is_authenticated:
            spotify_results = self.spotify_manager.search_tracks(query, limit=5)
            results.extend(spotify_results)
        
        # Search local library
        if self.local_manager.is_initialized:
            local_results = self.local_manager.search_local_music(query)
            results.extend(local_results[:5])  # Limit to 5 results
        
        if not results:
            return f"🔍 No results found for '{query}'"
        
        response = f"🔍 Found {len(results)} results for '{query}':\n"
        for i, song in enumerate(results[:5], 1):
            source_icon = "🎵" if song.source == "spotify" else "💿"
            response += f"{i}. {source_icon} {song.title} by {song.artist}"
            if song.album:
                response += f" ({song.album})"
            response += "\n"
        
        return response
    
    def _handle_playlist_request(self, playlist: str, user_input: str) -> str:
        """Handle playlist requests"""
        if not playlist:
            playlist = self._extract_playlist_from_input(user_input)
        
        if playlist:
            return self._play_playlist(playlist)
        else:
            return self._list_playlists()
    
    def _handle_status_request(self) -> str:
        """Handle status requests"""
        status = "🎵 Music Status:\n"
        
        # Spotify status
        if self.spotify_manager.is_authenticated:
            current_track = self.spotify_manager.get_current_track()
            if current_track:
                status += f"🎵 Spotify: {current_track.title} by {current_track.artist}\n"
            else:
                status += "🎵 Spotify: No track playing\n"
        else:
            status += "🎵 Spotify: Not connected\n"
        
        # Local status
        if self.local_manager.is_initialized:
            current_song = self.local_manager.get_current_song()
            if current_song:
                status += f"💿 Local: {current_song.title} by {current_song.artist}\n"
            else:
                status += "💿 Local: No music playing\n"
        else:
            status += "💿 Local: Not available\n"
        
        return status
    
    def _play_by_search(self, query: str) -> str:
        """Search and play music by query"""
        
        # Extract song and artist from query if possible
        song = ""
        artist = ""
        music_entities = {}
        
        # Try to parse "song by artist" format
        import re
        by_match = re.search(r'^(.*?)\s+by\s+(.+)$', query, re.IGNORECASE)
        if by_match:
            song = by_match.group(1).strip()
            artist = by_match.group(2).strip()
        
        # Check for platform preference in query
        prefer_youtube = any(platform in query.lower() for platform in ['youtube', 'yt'])
        prefer_spotify = any(platform in query.lower() for platform in ['spotify'])
        
        # Try YouTube Music first (most reliable, no API needed)
        if self.youtube_controller.is_available and (prefer_youtube or self.current_source in ["youtube", "auto"]):
            if song and artist:
                if self.youtube_controller.play_specific_song(song, artist):
                    return f"▶️ Opening '{song}' by {artist} on YouTube Music"
            else:
                if self.youtube_controller.search_and_play(query):
                    return f"▶️ Searching for '{query}' on YouTube Music"
        
        # Try desktop Spotify second (if running and not preferring YouTube)
        if not prefer_youtube and self.desktop_spotify.is_available and self.desktop_spotify.is_spotify_running():
            if self.desktop_spotify.search_and_play(query):
                return f"▶️ Searching and playing '{query}' on Spotify desktop"
        
        # Try Spotify Web API if available
        if not prefer_youtube and self.spotify_manager.is_authenticated and self.current_source in ["spotify", "auto"]:
            results = self.spotify_manager.search_tracks(query, limit=1)
            if results:
                song_result = results[0]
                if self.spotify_manager.play_track(song_result.uri):
                    return f"▶️ Playing '{song_result.title}' by {song_result.artist} on Spotify"
        
        # Try local files
        if self.local_manager.is_initialized and self.current_source in ["local", "auto"]:
            results = self.local_manager.search_local_music(query)
            if results:
                song_result = results[0]
                if self.local_manager.play_file(song_result.uri):
                    return f"▶️ Playing '{song_result.title}' by {song_result.artist} from local files"
        
        # Fallback: Open YouTube Music even if something failed
        if self.youtube_controller.is_available:
            if self.youtube_controller.search_and_play(query):
                return f"🎵 Opening YouTube Music search for '{query}'"
        
        # Last resort: Try to open Spotify
        if self.desktop_spotify.is_available:
            try:
                subprocess.Popen(["spotify"])
                return f"🎵 Opening Spotify... Please search for '{query}' manually"
            except:
                pass
        
        return f"❌ Unable to play '{query}'. Try:\n• Using YouTube Music (opens automatically)\n• Opening Spotify desktop app\n• Adding music files to ~/Music/"
    
    def _play_playlist(self, playlist_name: str) -> str:
        """Play a specific playlist"""
        if self.spotify_manager.is_authenticated:
            playlists = self.spotify_manager.get_user_playlists()
            for playlist in playlists:
                if playlist_name.lower() in playlist.name.lower():
                    tracks = self.spotify_manager.get_playlist_tracks(playlist.uri)
                    if tracks:
                        if self.spotify_manager.play_track(tracks[0].uri):
                            return f"▶️ Playing playlist '{playlist.name}' ({len(tracks)} songs)"
        
        return f"❌ Playlist '{playlist_name}' not found"
    
    def _list_playlists(self) -> str:
        """List available playlists"""
        if not self.spotify_manager.is_authenticated:
            return "🎵 Playlists require Spotify connection"
        
        playlists = self.spotify_manager.get_user_playlists()
        if not playlists:
            return "🎵 No playlists found"
        
        response = "🎵 Your playlists:\n"
        for i, playlist in enumerate(playlists[:10], 1):
            response += f"{i}. {playlist.name}\n"
        
        return response
    
    def _extract_search_terms(self, user_input: str) -> str:
        """Extract music search terms from user input"""
        # Remove common command words
        remove_words = {'play', 'music', 'song', 'track', 'some', 'the', 'a', 'an', 'by', 'from'}
        words = user_input.lower().split()
        search_words = [word for word in words if word not in remove_words]
        return ' '.join(search_words)
    
    def _extract_volume_from_input(self, user_input: str) -> Optional[int]:
        """Extract volume level from user input"""
        import re
        # Look for patterns like "50%", "volume 70", "to 80", etc.
        patterns = [
            r'(\d{1,3})%',
            r'volume (\d{1,3})',
            r'to (\d{1,3})',
            r'(\d{1,3})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                volume = int(match.group(1))
                return max(0, min(100, volume))  # Clamp between 0-100
        
        return None
    
    def _extract_playlist_from_input(self, user_input: str) -> str:
        """Extract playlist name from user input"""
        # Look for patterns like "play my workout playlist", "playlist called jazz"
        import re
        patterns = [
            r'playlist called (.+)',
            r'my (.+) playlist',
            r'playlist (.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1).strip()
        
        return ""
    
    def shutdown(self) -> None:
        """Cleanup plugin resources"""
        if self.local_manager.is_initialized:
            self.local_manager.stop()
        
        # Spotify cleanup is handled automatically by spotipy
        logger.info("Music plugin shutdown complete")
        pass

# Export the plugin class
__all__ = ['MusicPlugin']