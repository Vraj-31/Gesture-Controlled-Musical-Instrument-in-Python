"""
setup.py
--------
One-time setup:
  1. Downloads the MediaPipe HandLandmarker model file into the src/ folder.
  2. Checks that the audio backend (sounddevice + PortAudio) is actually
     working on this machine, since this is the #1 thing that silently
     fails on a fresh install -- pip installs sounddevice fine, but it
     needs the native PortAudio library too, which pip does NOT install
     for you on Linux (it usually does on Windows/Mac via wheels).

Usage:
    python setup.py
"""

import os
import sys
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[1/2] Model already exists at {MODEL_PATH}, skipping download.")
        return True

    print(f"[1/2] Downloading hand landmark model from:\n  {MODEL_URL}")
    print(f"      Saving to:\n  {MODEL_PATH}\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"      Done. Downloaded {size_kb:.0f} KB.")
        return True
    except Exception as e:
        print(f"      Download failed: {e}")
        print("      If this keeps failing (e.g. due to a firewall/proxy), download")
        print("      the file manually in a browser from the URL above and place it")
        print(f"      at: {MODEL_PATH}")
        return False


def check_audio_backend():
    print("\n[2/2] Checking audio backend (sounddevice + PortAudio)...")
    try:
        import sounddevice as sd
    except OSError:
        print("      PROBLEM: sounddevice is installed but its native PortAudio")
        print("      library is missing. Audio playback will NOT work until this")
        print("      is fixed (hand tracking will still work fine on its own).")
        print()
        if sys.platform.startswith("linux"):
            print("      Fix (Debian/Ubuntu):  sudo apt-get install libportaudio2")
            print("      Fix (Fedora):         sudo dnf install portaudio")
        elif sys.platform == "darwin":
            print("      Fix (macOS):          brew install portaudio")
        else:
            print("      Try reinstalling sounddevice:  pip install --force-reinstall sounddevice")
        return False

    try:
        devices = sd.query_devices()
        default_output = sd.default.device[1]
        print(f"      OK. Found {len(devices)} audio device(s).")
        if default_output is not None and default_output >= 0:
            print(f"      Default output device: {devices[default_output]['name']}")
        else:
            print("      WARNING: no default output device detected -- check your")
            print("      system sound settings if you don't hear anything later.")
        return True
    except Exception as e:
        print(f"      WARNING: could not query audio devices ({e}).")
        print("      Playback may still work, but couldn't be verified here.")
        return False


def main():
    model_ok = download_model()
    audio_ok = check_audio_backend()

    print("\n" + "=" * 60)
    if model_ok and audio_ok:
        print("Setup complete! You can now run:  python main_phase2.py")
    elif model_ok and not audio_ok:
        print("Hand tracking is ready, but audio needs attention (see above).")
        print("You can still run:  python main_phase2.py")
        print("(gestures and chord/strum detection will work; sound may not)")
    else:
        print("Setup incomplete -- see errors above before running main_phase2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

