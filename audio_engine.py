# ============================================================
# audio_engine.py — Audio Feedback for Salat Tracker
#
# Option 3: Pre-generate WAV files at startup using pyttsx3,
# then play them instantly during prayer using pygame.
#
# Why this solves the delay problem:
#   - pyttsx3 synthesis happens ONCE at startup, not during prayer
#   - pygame.mixer.Sound.play() is instant — no synthesis wait
#   - The user hears "Rakaat 2 of 4" the moment the rakaat completes
#
# Files generated at startup (saved to sounds/ folder):
#   rakaat_1.wav, rakaat_2.wav ...
#   rakaat_1_of_2.wav, rakaat_2_of_4.wav ...
#   complete_fajr.wav, complete_zuhr.wav ...
#   start_fajr_2.wav, start_zuhr_4.wav ...
#
# If pygame is not installed, falls back to original pyttsx3 TTS.
# ============================================================

import threading
import queue
import os
import time

# ── pygame ───────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[AudioEngine] pygame not installed — falling back to pyttsx3 TTS.")
    print("[AudioEngine] For instant playback: pip install pygame\n")

# ── pyttsx3 ──────────────────────────────────────────────────
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[AudioEngine] WARNING: pyttsx3 not installed.")
    print("[AudioEngine] Run:  pip install pyttsx3")
    print("[AudioEngine] Audio announcements will be disabled.\n")

SOUNDS_DIR       = "sounds"
PRAYER_NAMES     = ["Fajr", "Maghrib", "Zuhr", "Asr", "Isha", "Custom"]


class AudioEngine:
    """
    Pre-generates all announcement audio as WAV files at startup,
    then plays them instantly during prayer using pygame.

    Startup behaviour:
      1. Creates sounds/ folder
      2. Uses pyttsx3 to save each announcement as a WAV file
         (skips files that already exist from a previous run)
      3. Loads all WAV files into pygame Sound objects in memory

    During prayer:
      - announce_rakaat() / announce_complete() / announce_start()
        look up the pre-loaded Sound object and call .play() — instant

    Fallback:
      If pygame is not available, falls back to live pyttsx3 TTS
      (original behaviour with delay).
    """

    def __init__(self, rate=150, volume=0.9, enabled=True):
        self.enabled      = enabled
        self._rate        = rate
        self._volume      = volume
        self._sound_cache = {}   # key → pygame.mixer.Sound

        # TTS fallback
        self._queue      = queue.Queue()
        self._tts_thread = None

        if not self.enabled:
            return

        os.makedirs(SOUNDS_DIR, exist_ok=True)

        if PYGAME_AVAILABLE and PYTTSX3_AVAILABLE:
            print("[AudioEngine] Pre-generating audio files at startup...")
            self._pregenerate_all()
            print("[AudioEngine] Ready — instant playback enabled.")
        elif PYTTSX3_AVAILABLE:
            print("[AudioEngine] pygame unavailable — using TTS (with delay).")
            self._start_tts_worker()
        else:
            print("[AudioEngine] No audio library available.")
            self.enabled = False

    # ── Public API ───────────────────────────────────────────

    def announce_rakaat(self, count, total):
        """
        Plays instantly: "Rakaat 2 of 4" or "Rakaat 2"
        """
        if not self.enabled:
            return
        key = f"rakaat_{count}" if total >= 99 else f"rakaat_{count}_of_{total}"
        self._play(key)

    def announce_complete(self, prayer_name):
        """
        Plays instantly: "Fajr prayer complete. Alhamdulillah."
        """
        if not self.enabled:
            return
        self._play(f"complete_{prayer_name.lower()}")

    def announce_start(self, prayer_name, total_rakaat):
        """
        Plays instantly: "Starting Fajr prayer. 2 rakaat."
        """
        if not self.enabled:
            return
        if total_rakaat >= 99:
            key = f"start_{prayer_name.lower()}_unlimited"
        else:
            key = f"start_{prayer_name.lower()}_{total_rakaat}"
        self._play(key)

    def set_enabled(self, enabled):
        self.enabled = enabled
        print(f"[AudioEngine] Audio {'ON' if enabled else 'OFF'}")

    def shutdown(self):
        if self._tts_thread and self._tts_thread.is_alive():
            self._queue.put(None)
            self._tts_thread.join(timeout=3)
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ── Pre-generation ───────────────────────────────────────

    def _pregenerate_all(self):
        """Build text→key map, generate WAVs, load into cache."""
        texts = self._build_text_map()
        self._generate_wav_files(texts)
        self._load_sound_cache(texts)

    def _build_text_map(self):
        """Return dict of {cache_key: spoken_text} for all announcements."""
        texts = {}

        # Rakaat N (for unlimited prayers)
        for n in range(1, 11):
            texts[f"rakaat_{n}"] = f"Rakaat {n}"

        # Rakaat N of M (for fixed prayers — all valid combinations)
        for n in range(1, 5):
            for total in range(2, 5):
                if n <= total:
                    texts[f"rakaat_{n}_of_{total}"] = f"Rakaat {n} of {total}"

        # Prayer complete
        for name in PRAYER_NAMES:
            texts[f"complete_{name.lower()}"] = (
                f"{name} prayer complete. Alhamdulillah."
            )

        # Prayer start
        for name in PRAYER_NAMES:
            for n in range(2, 5):
                texts[f"start_{name.lower()}_{n}"] = (
                    f"Starting {name} prayer. {n} rakaat."
                )
            texts[f"start_{name.lower()}_unlimited"] = (
                f"Starting {name} prayer."
            )

        return texts

    def _generate_wav_files(self, texts):
        """
        Save each text as a WAV file using pyttsx3.
        Files that already exist are skipped so startup is fast
        after the first run.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate',   self._rate)
            engine.setProperty('volume', 1.0)   # max volume for file generation

            voices = engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    engine.setProperty('voice', v.id)
                    break

            generated = 0
            for key, text in texts.items():
                path = os.path.join(SOUNDS_DIR, f"{key}.wav")
                if not os.path.exists(path):
                    engine.save_to_file(text, path)
                    generated += 1

            if generated > 0:
                engine.runAndWait()
                print(f"[AudioEngine] Generated {generated} new WAV files "
                      f"in '{SOUNDS_DIR}/'")
            else:
                print(f"[AudioEngine] All WAV files already exist — "
                      f"skipping generation.")

            engine.stop()

        except Exception as e:
            print(f"[AudioEngine] WAV generation error: {e}")
            print("[AudioEngine] Switching to live TTS fallback.")
            if not self._tts_thread:
                self._start_tts_worker()

    def _load_sound_cache(self, texts):
        """Load all WAV files into pygame Sound objects."""
        loaded = 0
        missing = 0
        for key in texts:
            path = os.path.join(SOUNDS_DIR, f"{key}.wav")
            if os.path.exists(path):
                try:
                    sound = pygame.mixer.Sound(path)
                    sound.set_volume(self._volume)
                    self._sound_cache[key] = sound
                    loaded += 1
                except Exception as e:
                    print(f"[AudioEngine] Could not load {key}.wav: {e}")
                    missing += 1
            else:
                missing += 1

        print(f"[AudioEngine] {loaded} sounds loaded into memory"
              + (f" ({missing} missing)" if missing else "") + ".")

    # ── Playback ─────────────────────────────────────────────

    def _play(self, key):
        """
        Play a sound instantly from cache.
        Falls back to live TTS if the key is not in cache.
        """
        if PYGAME_AVAILABLE and key in self._sound_cache:
            # Stop any currently playing announcement to avoid overlap
            self._sound_cache[key].stop()
            self._sound_cache[key].play()
        else:
            # TTS fallback
            text = self._key_to_text(key)
            if text and PYTTSX3_AVAILABLE:
                if not self._tts_thread:
                    self._start_tts_worker()
                self._enqueue(text)

    def _key_to_text(self, key):
        """Convert cache key back to spoken text for TTS fallback."""
        try:
            p = key.split("_")
            if p[0] == "rakaat":
                if "of" in p:
                    return f"Rakaat {p[1]} of {p[3]}"
                return f"Rakaat {p[1]}"
            if p[0] == "complete":
                return f"{p[1].capitalize()} prayer complete. Alhamdulillah."
            if p[0] == "start":
                name = p[1].capitalize()
                return (f"Starting {name} prayer."
                        if p[-1] == "unlimited"
                        else f"Starting {name} prayer. {p[-1]} rakaat.")
        except Exception:
            pass
        return None

    # ── TTS fallback worker ──────────────────────────────────

    def _enqueue(self, text):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def _start_tts_worker(self):
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTSWorker"
        )
        self._tts_thread.start()

    def _tts_worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate',   self._rate)
            engine.setProperty('volume', self._volume)
            voices = engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    engine.setProperty('voice', v.id)
                    break
        except Exception as e:
            print(f"[AudioEngine] TTS init failed: {e}")
            return

        while True:
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[AudioEngine] TTS error: {e}")
        try:
            engine.stop()
        except Exception:
            pass