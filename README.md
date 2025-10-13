# Whisper Daily

**Offline [Whisper](https://github.com/openai/whisper)-powered transcription for live audio, files, and subtitle generation — with optional VAD and auto audio‑routing.**

Whisper Daily is a single Python script that can:
- **Live transcribe** your system audio or microphone.
- **Transcribe files** (`--file`) and **generate SRT subtitles** (`--srt`).
- **Auto‑switch audio routing** to a virtual cable (`--cable`) or **Voicemeeter** (`--voicemeeter`, Windows only).
- Optionally use **VAD (webrtcvad)** to suppress silence and print cleaner text.
- Print **timestamps** for sentence starts and do **on-the-fly correction** of draft text.

> **Virtual audio device is recommended** if you want to capture *system audio* (YouTube, Zoom, etc.).  
> Use **VB‑CABLE** on Windows/macOS, or **Voicemeeter Banana** on Windows. Microphone-only capture can work without a virtual device, but `--cable`/`--voicemeeter` makes system routing painless.

---

## Table of Contents
- [Why a virtual audio device?](#why-a-virtual-audio-device)
- [Installation](#installation)
  - [Common prerequisites](#common-prerequisites)
  - [Windows setup](#windows-setup)
  - [macOS setup](#macos-setup)
- [Quick start](#quick-start)
- [CLI options](#cli-options)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

---

## Why a virtual audio device?
Operating systems don’t expose “what you hear” (system mix) as a normal microphone.  
A **virtual audio device** solves that by acting like a **playback sink** (you set system output to it) and a **recording source** (the script reads from it).

- **VB‑CABLE** (Windows/macOS): simple and lightweight.
- **Voicemeeter Banana** (Windows): advanced routing, separate *Audio* vs *Communications* outputs, EQ, gain, etc.

`--cable` switches your **system playback** to the *CABLE Input* and listens from *CABLE Output*.  
`--voicemeeter` (Windows) starts/uses Voicemeeter, sets **Voicemeeter Input** (Audio) and **Voicemeeter AUX Input** (Communications), and records from **Voicemeeter Out B1**.

---

## Installation

### Common prerequisites
1. **Python 3.9+** recommended.
2. **FFmpeg** (required by Whisper):
   - Windows: Download static build and add `ffmpeg.exe` to PATH.
   - macOS (Homebrew): `brew install ffmpeg`
3. Python deps:
   ```bash
   pip install openai-whisper sounddevice soundfile numpy webrtcvad
   ```
   > If you only need file transcription (no live/VAD), you can skip `sounddevice`, `webrtcvad`.

### Windows setup
> Choose **one**: VB‑CABLE *(simpler)* or Voicemeeter Banana *(advanced)*.

**Option A — VB‑CABLE (recommended for quick start)**
1. Download and install **VB‑CABLE** from VB‑Audio.
2. Reboot if prompted.
3. (First run) The script can automatically set **system playback** to *CABLE Input* when using `--cable`.

**Option B — Voicemeeter Banana (advanced)**
1. Install **Voicemeeter Banana** from VB‑Audio.
2. Optional: prepare your `.xml` configs (e.g., `buds3.xml` / `desktop.xml`) if you want auto‑profiles.
3. Install PowerShell module (used to switch default devices):
   ```powershell
   Install-Module -Name AudioDeviceCmdlets -Force
   ```
4. Run with `--voicemeeter` to auto‑switch **Audio** to *Voicemeeter Input* and **Communications** to *Voicemeeter AUX Input*.
   The script records from *Voicemeeter Out B1*.

> The script will check/install `AudioDeviceCmdlets` on first run if missing.

### macOS setup
**Virtual device (one of):**
- **VB‑CABLE for Mac** (install the macOS package from VB‑Audio).
- *(Voicemeeter is Windows‑only)*

**Audio switching helper (required for `--cable`):**
```bash
brew install switchaudio-osx
```
The script uses `SwitchAudioSource` to change the **default output** to *CABLE Input* during the session and restore it afterward.

---

## Quick start

```bash
# Live transcription of system audio via VB‑CABLE
python whisper_daily.py --cable --model base --language auto --vad

# Transcribe a media file and print text
python whisper_daily.py --file data/sample.mp3 --model small

# Create SRT subtitles
python whisper_daily.py --srt data/lecture.mp4 --model small
```

- List capture devices (useful if you prefer a specific mic):
  ```bash
  python whisper_daily.py --list-devices
  ```

---

## CLI options

- `--file <path>`: transcribe the file and exit.
- `--srt <path>`: generate `file.srt` from the media.
- `--cable`: **Auto** set system **playback** to *CABLE Input* (Windows/macOS). Ignores `--device`.
- `--voicemeeter`: **Windows only**. Start/configure Voicemeeter, set **Audio** to *Voicemeeter Input* and **Communications** to *Voicemeeter AUX Input*. Ignores `--device`.
- `--no-control`: when used with `--cable`, skips opening the OS audio panel.
- `--device <name|index>`: choose input device (when not using `--cable`/`--voicemeeter`).
- `--model <name>`: Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` (default: `base`).
- `--language <code|auto>`: force language (e.g., `en`, `ru`) or let Whisper detect via `auto`.
- `--sr <int>`: sample rate for live mode (default `16000`).
- `--block-sec <float>`: block size for live streaming (default `1.0`).
- `--channels <int>`: 0=auto (<=2) or explicit number of input channels.
- `--translate`: translate to English instead of transcribing.
- `--list-devices`: list input devices and exit.
- **VAD**: `--vad` to enable; `--vad-aggressiveness {0..3}` (default `1`).

Notes:
- On Apple Silicon, the script tries to use **MPS** (Metal) if available; otherwise CPU.
- Whisper parameters: low temperature, word timestamps disabled in live mode for speed, etc.

---

## Examples

**Windows + VB‑CABLE (system audio):**
```bash
python whisper_daily.py --cable --model small --language auto --vad
```

**Windows + Voicemeeter (separate Audio/Comms):**
```bash
python whisper_daily.py --voicemeeter --model small --language auto --vad
```

**macOS + VB‑CABLE:**
```bash
python whisper_daily.py --cable --model base --language auto
```

**Mic only (no virtual device):**
```bash
python whisper_daily.py --device "USB Microphone" --model base
```

**File transcription with timestamps for sentence starts:**
```bash
python whisper_daily.py --file path/to/talk.wav --model small --language auto
```

**SRT generation with smart splitting:**
```bash
python whisper_daily.py --srt path/to/lecture.mp4 --model small
```

---

## Troubleshooting

- **“VB‑CABLE is not fully installed!”**  
  Ensure both *CABLE Input* (playback) and *CABLE Output* (recording) devices exist in OS sound settings. Reboot after driver install.

- **No text printed / empty transcription**  
  Check FFmpeg presence, try a smaller model (`--model base`), ensure audio is actually routed into the selected input.

- **PortAudio / device unavailable**  
  Use `--list-devices`, pick a valid index with `--device`, or close apps that occupy the device (DAWs, Zoom).

- **macOS: `SwitchAudioSource` not found**  
  `brew install switchaudio-osx`

- **Windows: AudioDeviceCmdlets not found**  
  Run PowerShell as admin once and install:
  ```powershell
  Install-Module -Name AudioDeviceCmdlets -Force
  ```

- **Bluetooth earbuds profile changes / poor quality**  
  Some BT stacks switch between A2DP/HFP. For best quality, avoid mic‑over‑Bluetooth while recording system audio; prefer wired output or Voicemeeter routing.

---

## FAQ

**Q: Do I need VB‑CABLE / Voicemeeter for microphone capture?**  
A: No. You can select your mic with `--device`. Virtual devices are only needed to capture **system audio**.

**Q: Can I use this on macOS without Voicemeeter?**  
A: Yes. Voicemeeter is Windows‑only. On macOS use **VB‑CABLE** and `SwitchAudioSource` (installed via Homebrew) with `--cable`.

**Q: Which Whisper model should I pick?**  
A: Start with `base` or `small`. Larger models are more accurate but slower. Apple Silicon can use **MPS** automatically.

**Q: What does VAD do?**  
A: `--vad` uses webrtcvad to suppress silence, show cleaner partial text, and trigger “final” corrections after speech ends.

---

## License
MIT (see `LICENSE` in the repo).
