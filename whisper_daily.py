#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper (offline) live STT + File mode + Subtitle generation with VAD:
- --file <path> : transcribe file and exit
- --srt <path>  : create subtitles from file
- --cable       : auto-switch system audio devices
- --device      : index/substring of input device (when --cable is not used)
- --vad         : enable Voice Activity Detection to filter silence

Dependencies:
  pip install openai-whisper sounddevice soundfile numpy webrtcvad
And make sure FFmpeg is installed on the system.

On Windows, the --cable mode requires the PowerShell module AudioDeviceCmdlets:
  Install-Module -Name AudioDeviceCmdlets -Force

On macOS, the --cable mode requires switchaudio-osx:
  brew install switchaudio-osx
"""

import argparse
import os
import sys
import time
import platform
import logging
import subprocess
import re
import json
import shutil
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import sounddevice as sd

# --- Optional PyTorch/MPS detection for Apple Silicon ---
# Enable safe fallback on MPS if an op is not implemented
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch  # type: ignore
    MPS_AVAILABLE = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except Exception:
    torch = None
    MPS_AVAILABLE = False

def pick_infer_device() -> str:
    # Prefer MPS on macOS if available; otherwise CPU
    if IS_MAC and MPS_AVAILABLE:
        return "mps"
    return "cpu"

# --- VAD (optional) ---
VAD_AVAILABLE = False
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None

# --- Whisper: verify correct package ---
import whisper as _whisper
if not hasattr(_whisper, "load_model"):
    raise RuntimeError("Incorrect 'whisper' imported. Install 'openai-whisper' and do not name your file 'whisper.py'.")
whisper = _whisper

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_audio.log', encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Detect platform
IS_WIN = platform.system().lower().startswith("win")
IS_MAC = platform.system().lower() == "darwin"

# Regex to find VB-CABLE devices
RECORD_REGEX = r".*CABLE\s*Output.*VB-Audio.*"     # For recording look for CABLE Output
PLAYBACK_REGEX = r".*CABLE\s*Input.*VB-Audio.*"    # For playback look for CABLE Input
VOICEMEETER_RECORD_REGEX = r".*Out\s+B1.*"                        # For recording
VOICEMEETER_PLAYBACK_AUDIO_REGEX = r".*Voicemeeter\s+Input.*"     # For playback (Audio)
VOICEMEETER_PLAYBACK_COMM_REGEX = r".*Voicemeeter\s+AUX\s+Input.*"  # For playback (Communications)
HEADPHONES_REGEX = r".*Buds3.*"                                   # To detect headphones
VOICEMEETER_PATH = "C:\\Program Files (x86)\\VB\\Voicemeeter"

# ------------------------------------------------------------
#  PowerShell audio switcher
# ------------------------------------------------------------
def get_current_default_devices_macos():
    """Get current default devices on macOS"""
    try:
        # Get output device
        result_output = subprocess.run([
            'osascript', '-e', 
            'tell application "System Events" to get the name of (output volume of (get volume settings))'
        ], capture_output=True, text=True, timeout=15)
        
        # Alternative via system_profiler for more details
        result_audio = subprocess.run([
            'system_profiler', 'SPAudioDataType', '-json'
        ], capture_output=True, text=True, timeout=15)
        
        output_device = None
        if result_audio.returncode == 0:
            try:
                audio_data = json.loads(result_audio.stdout)
                for item in audio_data.get('SPAudioDataType', []):
                    if 'default_output_device' in str(item).lower() or item.get('_name', '').lower() == 'built-in output':
                        # Try to infer current output device in the system
                        pass
            except json.JSONDecodeError:
                pass
        
        # Use SwitchAudioSource if available
        try:
            result_current = subprocess.run([
                'SwitchAudioSource', '-c'
            ], capture_output=True, text=True, timeout=10)
            if result_current.returncode == 0:
                output_device = result_current.stdout.strip()
        except FileNotFoundError:
            # SwitchAudioSource not installed
            pass
            
        return output_device, None  # Return only output, input remains unchanged
        
    except Exception as e:
        logger.warning(f"Error getting current macOS devices: {e}")
        return None, None

def check_audiocmdlets_module():
    """Check if AudioDeviceCmdlets module is available"""
    try:
        ps_script = '''
        if (Get-Module -ListAvailable -Name AudioDeviceCmdlets) {
            Write-Host "OK"
        } else {
            Write-Host "NOT_FOUND"
        }
        '''
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, timeout=10)
        
        return "OK" in result.stdout
    except Exception:
        return False

def install_audiocmdlets_module():
    """Install AudioDeviceCmdlets module"""
    print("Installing AudioDeviceCmdlets module...")
    try:
        ps_script = '''
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Install-Module -Name AudioDeviceCmdlets -Force -AllowClobber
        Write-Host "Module installed successfully"
        '''
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("AudioDeviceCmdlets module installed")
            return True
        else:
            print(f"Installation error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Module installation error: {e}")
        return False

def get_current_default_devices_windows():
    """Get current default devices via PowerShell (Windows)"""
    try:
        ps_script = '''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        
        $playback = Get-AudioDevice -Playback
        $recording = Get-AudioDevice -Recording
        
        Write-Host "PLAYBACK:$($playback.Name)|$($playback.ID)|$($playback.Type)"
        '''
        
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            logger.warning(f"Failed to get current devices: {result.stderr}")
            return None, None
            
        playback_info = None
        
        for line in result.stdout.split('\n'):
            if line.startswith('PLAYBACK:'):
                playback_info = line[9:].strip()  # Remove "PLAYBACK:"
                
        return playback_info, None
        
    except Exception as e:
        logger.warning(f"Error getting current devices: {e}")
        return None, None

def set_default_audio_device(device_name: str, device_type: str):
    """
    Set default audio device cross-platform
    device_type: only "playback" (recording not supported in --cable mode)
    """
    if device_type != "playback":
        logger.warning(f"Only playback device is supported in --cable mode")
        return False
        
    if IS_WIN:
        return _set_default_audio_device_windows(device_name)
    elif IS_MAC:
        return _set_default_audio_device_macos(device_name)
    else:
        logger.error("Unsupported platform for --cable mode")
        return False

def _set_default_audio_device_windows(device_name: str):
    """Set default audio device on Windows"""
    try:
        ps_script = f'''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        
        $device = Get-AudioDevice -List | Where-Object {{$_.Name -like "*{device_name}*" -and $_.Type -eq "Playback"}}
        
        if ($device) {{
            # Set both Audio and Communications
            Set-AudioDevice -ID $device.ID
            Set-AudioDevice -ID $device.ID -CommunicationDevice
            Write-Host "SUCCESS:$($device.Name)"
        }} else {{
            Write-Host "ERROR:Device not found"
            exit 1
        }}
        '''
        
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, encoding='utf-8', timeout=30)
        
        if result.returncode == 0 and "SUCCESS:" in result.stdout:
            device_set = result.stdout.split("SUCCESS:")[1].strip()
            logger.info(f"Playback set (Audio+Communications): {device_set}")
            return True
        else:
            logger.error(f"Playback set error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"PowerShell playback set error: {e}")
        return False

def _set_default_audio_device_windows_dual(audio_device_name: str, comm_device_name: str):
    """Set different default playback devices for Audio and Communications on Windows"""
    try:
        ps_script = f'''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        
        $audioDevice = Get-AudioDevice -List | Where-Object {{$_.Name -like "*{audio_device_name}*" -and $_.Type -eq "Playback"}}
        $commDevice = Get-AudioDevice -List | Where-Object {{$_.Name -like "*{comm_device_name}*" -and $_.Type -eq "Playback"}}
        
        $success = $true
        
        if ($audioDevice) {{
            Set-AudioDevice -ID $audioDevice.ID
            Write-Host "AUDIO_SUCCESS:$($audioDevice.Name)"
        }} else {{
            Write-Host "AUDIO_ERROR:Device not found"
            $success = $false
        }}
        
        if ($commDevice) {{
            Set-AudioDevice -ID $commDevice.ID -CommunicationDevice
            Write-Host "COMM_SUCCESS:$($commDevice.Name)"
        }} else {{
            Write-Host "COMM_ERROR:Device not found"
            $success = $false
        }}
        
        if (-not $success) {{
            exit 1
        }}
        '''
        
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, encoding='utf-8', timeout=30)
        
        if result.returncode == 0 and "AUDIO_SUCCESS:" in result.stdout and "COMM_SUCCESS:" in result.stdout:
            audio_set = result.stdout.split("AUDIO_SUCCESS:")[1].split("\n")[0].strip()
            comm_set = result.stdout.split("COMM_SUCCESS:")[1].split("\n")[0].strip()
            logger.info(f"Audio set: {audio_set}")
            logger.info(f"Communications set: {comm_set}")
            return True
        else:
            logger.error(f"Device set error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"PowerShell dual device set error: {e}")
        return False


def _set_default_audio_device_macos(device_name: str):
    """Set default audio device on macOS"""
    try:
        # Ensure SwitchAudioSource exists
        try:
            subprocess.run(['SwitchAudioSource', '-h'], 
                         capture_output=True, timeout=5, check=False)
        except FileNotFoundError:
            logger.error("SwitchAudioSource is not installed. Install with: brew install switchaudio-osx")
            return False
        
        # Get device list
        result_list = subprocess.run([
            'SwitchAudioSource', '-a'
        ], capture_output=True, text=True, timeout=15)
        
        if result_list.returncode != 0:
            logger.error("Failed to get audio device list")
            return False
        
        # Find device by name
        devices = result_list.stdout.strip().split('\n')
        target_device = None
        
        for device in devices:
            if device_name.lower() in device.lower():
                target_device = device.strip()
                break
        
        if not target_device:
            logger.error(f"Device '{device_name}' not found in available list")
            return False
        
        # Set device
        result_set = subprocess.run([
            'SwitchAudioSource', '-s', target_device
        ], capture_output=True, text=True, timeout=15)
        
        if result_set.returncode == 0:
            logger.info(f"Playback set: {target_device}")
            return True
        else:
            logger.error(f"Device set error: {result_set.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Audio device set error on macOS: {e}")
        return False

def get_current_default_devices_windows_dual():
    """Get current default Audio and Communications playback devices via PowerShell (Windows)"""
    try:
        ps_script = '''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        
        $playback = Get-AudioDevice -Playback
        $communication = Get-AudioDevice -Communication
        
        Write-Host "PLAYBACK:$($playback.Name)|$($playback.ID)|$($playback.Type)"
        Write-Host "COMMUNICATION:$($communication.Name)|$($communication.ID)|$($communication.Type)"
        '''
        
        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            logger.warning(f"Failed to get current devices: {result.stderr}")
            return None, None
            
        playback_info = None
        communication_info = None
        
        for line in result.stdout.split('\n'):
            if line.startswith('PLAYBACK:'):
                playback_info = line[9:].strip()  # Remove "PLAYBACK:"
            elif line.startswith('COMMUNICATION:'):
                communication_info = line[13:].strip()  # Remove "COMMUNICATION:"
                
        return playback_info, communication_info
        
    except Exception as e:
        logger.warning(f"Error getting current dual devices: {e}")
        return None, None

def check_bluetooth_headphones_connected():
    """Check if bluetooth headphones are connected"""
    headphones_index = find_output_index_by_regex(HEADPHONES_REGEX)
    return headphones_index is not None

def restore_audio_device(device_info: str, device_type: str):
    """Restore device cross-platform"""
    if device_type != "playback":
        return True  # In the new mode we only restore playback
        
    if IS_WIN:
        return _restore_audio_device_windows(device_info)
    elif IS_MAC:
        return _restore_audio_device_macos(device_info)
    else:
        logger.error("Unsupported platform for audio restore")
        return False

def restore_audio_devices_dual(audio_device_info: str, comm_device_info: str):
    """Restore separate devices for Audio and Communications"""
    if IS_WIN:
        return _restore_audio_devices_windows_dual(audio_device_info, comm_device_info)
    elif IS_MAC:
        # On macOS we only use the audio device
        return _restore_audio_device_macos(audio_device_info)
    else:
        logger.error("Unsupported platform for dual audio restore")
        return False

def _restore_audio_devices_windows_dual(audio_device_info: str, comm_device_info: str):
    """Restore separate devices on Windows"""
    success = True
    if audio_device_info:
        success &= _restore_audio_device_windows(audio_device_info)
    if comm_device_info:
        # Separate logic for communications device
        success &= _restore_communication_device_windows(comm_device_info)
    return success

def _restore_audio_device_windows(device_info: str):
    """Restore Windows device from info string 'Name|ID|Type'"""
    if not device_info or '|' not in device_info:
        return False
        
    try:
        parts = device_info.split('|')
        device_name = parts[0]
        device_id = parts[1] if len(parts) > 1 else parts[0]

        ps_script = f'''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        # Restore for both Audio and Communications
        Set-AudioDevice -ID "{device_id}"
        Set-AudioDevice -ID "{device_id}" -CommunicationDevice
        Write-Host "RESTORED:{device_name}"
        '''

        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, encoding='utf-8', timeout=30)
        
        if result.returncode == 0:
            logger.info(f"Playback restored (Audio+Communications): {device_name}")
            return True
        else:
            logger.error(f"Playback restore error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Playback restore exception: {e}")
        return False

def _restore_communication_device_windows(device_info: str):
    """Restore Windows communications device"""
    if not device_info or '|' not in device_info:
        return False
        
    try:
        parts = device_info.split('|')
        device_name = parts[0]
        device_id = parts[1] if len(parts) > 1 else parts[0]

        ps_script = f'''
        Import-Module AudioDeviceCmdlets -ErrorAction Stop
        Set-AudioDevice -ID "{device_id}" -CommunicationDevice
        Write-Host "RESTORED_COMM:{device_name}"
        '''

        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script
        ], capture_output=True, text=True, encoding='utf-8', timeout=30)
        
        if result.returncode == 0:
            logger.info(f"Communications restored: {device_name}")
            return True
        else:
            logger.error(f"Communications restore error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Communications restore exception: {e}")
        return False

def _restore_audio_device_macos(device_info: str):
    """Restore macOS device"""
    if not device_info:
        return False
        
    try:
        device_name = device_info.strip()
        
        result = subprocess.run([
            'SwitchAudioSource', '-s', device_name
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            logger.info(f"Playback restored: {device_name}")
            return True
        else:
            logger.error(f"Device restore error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Audio device restore error on macOS: {e}")
        return False

def start_voicemeeter_with_config(config_name: str):
    """Start Voicemeeter with the specified config"""
    if not IS_WIN:
        logger.warning("Voicemeeter is available on Windows only")
        return False
        
    try:
        # Try to find Voicemeeter in standard locations
        possible_paths = [VOICEMEETER_PATH]
        
        voicemeeter_exe = None
        for path in possible_paths:
            if os.path.exists(path):
                voicemeeter_exe = path
                break
                
        if not voicemeeter_exe:
            logger.error("Voicemeeter not found. Check your installation.")
            return False
            
        # Launch with config
        subprocess.Popen(f'voicemeeterpro.exe -L"{config_name}"', shell=True, cwd=VOICEMEETER_PATH)
        time.sleep(5)  # Give it time to start
        logger.info(f"Voicemeeter started with config: {config_name}")
        return True
        
    except Exception as e:
        logger.error(f"Voicemeeter start error: {e}")
        return False

def shutdown_voicemeeter():
    """Terminate Voicemeeter processes cross-platform"""
    if IS_WIN:
        return _shutdown_voicemeeter_windows()
    elif IS_MAC:
        return _shutdown_voicemeeter_macos()
    else:
        logger.warning("Voicemeeter shutdown is not supported on this platform")
        return False

def _shutdown_voicemeeter_windows():
    """Terminate Voicemeeter on Windows"""
    try:
        # Possible process names
        voicemeeter_processes = [
            "voicemeeterpro.exe",
        ]
        
        terminated_processes = []
        
        for process_name in voicemeeter_processes:
            try:
                # Try to kill via taskkill
                result = subprocess.run([
                    "taskkill", "/F", "/IM", process_name
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    terminated_processes.append(process_name)
                    logger.info(f"Terminated process: {process_name}")
                    
            except Exception as e:
                logger.debug(f"Could not terminate {process_name}: {e}")
                continue
        
        if terminated_processes:
            logger.info(f"Voicemeeter processes terminated: {', '.join(terminated_processes)}")
            time.sleep(1)  # Let it finish
            return True
        else:
            logger.debug("Voicemeeter processes not found or already closed")
            return True
            
    except Exception as e:
        logger.error(f"Voicemeeter shutdown error on Windows: {e}")
        return False

def _shutdown_voicemeeter_macos():
    """Terminate Voicemeeter on macOS (if running via Wine or similar)"""
    try:
        # On macOS Voicemeeter might run via Wine
        voicemeeter_processes = [
            "voicemeeterpro.exe",
        ]
        
        terminated_processes = []
        
        # Use pkill to find and terminate processes
        for process_name in voicemeeter_processes:
            try:
                # Search by name with and without extension
                process_base = process_name.replace('.exe', '')
                
                for name_variant in [process_name, process_base]:
                    result = subprocess.run([
                        "pkill", "-f", name_variant
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        terminated_processes.append(name_variant)
                        logger.info(f"Terminated process: {name_variant}")
                        
            except Exception as e:
                logger.debug(f"Could not terminate {process_name}: {e}")
                continue
        
        if terminated_processes:
            logger.info(f"Voicemeeter processes terminated: {', '.join(terminated_processes)}")
            time.sleep(1)  # Let it finish
            return True
        else:
            logger.debug("Voicemeeter processes not found or already closed")
            return True
            
    except Exception as e:
        logger.error(f"Voicemeeter shutdown error on macOS: {e}")
        return False

def open_audio_settings():
    """Open audio settings panel cross-platform"""
    try:
        if IS_WIN:
            # Windows: open sound settings
            control = shutil.which("control.exe")
            if not control:
                control = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "control.exe")
            subprocess.Popen([control, "mmsys.cpl,,1"], close_fds=True)
        elif IS_MAC:
            # macOS: open system sound preferences
            subprocess.Popen(['open', '/System/Library/PreferencePanes/Sound.prefPane'], 
                           close_fds=True)
        else:
            logger.warning("Opening audio settings is not supported on this platform")
    except Exception as e:
        logger.error(f"Error opening audio settings panel: {e}")
    

class VoicemeeterSwitcher:
    """Audio device switcher for Voicemeeter"""
    
    def __init__(self):
        self.prev_audio_device = None
        self.prev_comm_device = None
        self.switched_successfully = False
        self.voicemeeter_started = False

    def __enter__(self):
        if not (IS_WIN or IS_MAC):
            raise RuntimeError("Voicemeeter switching is available on Windows and macOS only")
        
        if IS_WIN:
            return self._enter_windows()
        elif IS_MAC:
            return self._enter_macos()
    
    def _enter_windows(self):
        """Initialization for Windows"""
        # Check and install module if needed
        if not check_audiocmdlets_module():
            print("AudioDeviceCmdlets module not found")
            if input("Install the module? (y/n): ").lower().startswith('y'):
                if not install_audiocmdlets_module():
                    raise RuntimeError("Failed to install AudioDeviceCmdlets")
            else:
                raise RuntimeError("AudioDeviceCmdlets is required for --voicemeeter mode")

        print("\n=== Voicemeeter setup (Windows) ===")
        
        # 1. Check headphones connection
        headphones_connected = check_bluetooth_headphones_connected()
        logger.info(f"Bluetooth headphones: {'connected' if headphones_connected else 'not connected'}")
        
        # 2. Start Voicemeeter with appropriate config
        config_name = "buds3.xml" if headphones_connected else "desktop.xml"
        logger.info(f"Loading config: {config_name}")
        
        if not start_voicemeeter_with_config(config_name):
            logger.warning("Failed to start Voicemeeter with config")
        else:
            self.voicemeeter_started = True
        
        # 3. Save current devices
        self.prev_audio_device, self.prev_comm_device = get_current_default_devices_windows_dual()
        
        if self.prev_audio_device:
            logger.info(f"Current Audio: {self.prev_audio_device.split('|')[0]}")
        if self.prev_comm_device:
            logger.info(f"Current Communications: {self.prev_comm_device.split('|')[0]}")
        
        # 4. Set Voicemeeter devices
        audio_device = "Voicemeeter Input"
        comm_device = "Voicemeeter AUX Input"
        
        success = _set_default_audio_device_windows_dual(audio_device, comm_device)
        
        if success:
            print("Devices switched to Voicemeeter")
            self.switched_successfully = True
        else:
            print("Switching failed. Check Voicemeeter installation")
            
        return self
    
    def _enter_macos(self):
        """Initialization for macOS"""
        print("\n=== Voicemeeter setup (macOS) ===")
        logger.warning("Voicemeeter is not available on macOS. Using alternative mode.")
        
        # On macOS act like regular CABLE mode but with different devices
        self.prev_audio_device, _ = get_current_default_devices_macos()
        
        if self.prev_audio_device:
            logger.info(f"Current playback: {self.prev_audio_device}")
        
        # Try to set Voicemeeter device or fallback to CABLE
        audio_ok = set_default_audio_device("Voicemeeter Input", "playback")
        if not audio_ok:
            # Fallback to CABLE if Voicemeeter not found
            audio_ok = set_default_audio_device("CABLE Input", "playback")
        
        self.switched_successfully = audio_ok
        
        if self.switched_successfully:
            print("Playback device switched")
        else:
            print("Switching failed")
            
        return self        

    def __exit__(self, exc_type, exc, tb):
        if self.switched_successfully:
            print("\n=== Restoring audio devices ===")
            
            if IS_WIN and self.prev_audio_device and self.prev_comm_device:
                restore_audio_devices_dual(self.prev_audio_device, self.prev_comm_device)
            elif IS_MAC and self.prev_audio_device:
                restore_audio_device(self.prev_audio_device, "playback")
        
        # Terminate Voicemeeter if we started it
        if self.voicemeeter_started:
            logger.info("Shutting down Voicemeeter...")
            if shutdown_voicemeeter():
                logger.info("Voicemeeter shut down successfully")
        return False

class AudioSwitcher:
    """Audio device switcher"""
    
    def __init__(self):
        self.prev_playback = None
        self.switched_successfully = False

    def __enter__(self):
        if not (IS_WIN or IS_MAC):
            raise RuntimeError("--cable audio switching is available on Windows and macOS only")
        
        if IS_WIN:
            return self._enter_windows()
        elif IS_MAC:
            return self._enter_macos()
    
    def _enter_windows(self):
        """Initialization for Windows"""
        # Check and install module if needed
        if not check_audiocmdlets_module():
            print("AudioDeviceCmdlets module not found")
            if input("Install the module? (y/n): ").lower().startswith('y'):
                if not install_audiocmdlets_module():
                    raise RuntimeError("Failed to install AudioDeviceCmdlets")
            else:
                raise RuntimeError("AudioDeviceCmdlets is required for --cable mode")

        print("\n=== Audio device switching (Windows) ===")
        
        # Save current devices
        self.prev_playback, _ = get_current_default_devices_windows()
        
        if self.prev_playback:
            logger.info(f"Current playback: {self.prev_playback.split('|')[0]}")
        
        # Set CABLE device for output only
        playback_ok = set_default_audio_device("CABLE Input", "playback")
        
        if not playback_ok:
            logger.warning("Failed to set CABLE Input for playback")
             
        self.switched_successfully = playback_ok
        
        if self.switched_successfully:
            print("Playback device switched to CABLE Input")
        else:
            print("Switching failed. Check VB-CABLE installation")
            
        return self
    
    def _enter_macos(self):
        """Initialization for macOS"""
        print("\n=== Audio device switching (macOS) ===")
        
        # Ensure SwitchAudioSource exists
        try:
            subprocess.run(['SwitchAudioSource', '-h'], 
                         capture_output=True, timeout=5, check=False)
        except FileNotFoundError:
            print("SwitchAudioSource is required for --cable mode on macOS")
            if input("Install via Homebrew? (brew install switchaudio-osx) (y/n): ").lower().startswith('y'):
                try:
                    subprocess.run(['brew', 'install', 'switchaudio-osx'], 
                                 timeout=120, check=True)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Installation error: {e}")
                    raise RuntimeError("Install SwitchAudioSource: brew install switchaudio-osx")
            else:
                raise RuntimeError("SwitchAudioSource is required for --cable mode on macOS")

        # Save current device
        self.prev_playback, _ = get_current_default_devices_macos()
        
        if self.prev_playback:
            logger.info(f"Current playback: {self.prev_playback}")
        
        # Set CABLE device
        playback_ok = set_default_audio_device("CABLE Input", "playback") 
        
        if not playback_ok:
            logger.warning("Failed to set CABLE Input for playback")
            
        self.switched_successfully = playback_ok
        
        if self.switched_successfully:
            print("Playback device switched to CABLE Input")
        else:
            print("Switching failed. Check VB-CABLE installation")
            
        return self        

    def __exit__(self, exc_type, exc, tb):
        if self.switched_successfully:
            print("\n=== Restoring playback device ===")
            
            if self.prev_playback:
                restore_audio_device(self.prev_playback, "playback")
                
        return False

# VB-CABLE check functions
def check_vb_cable_installed():
    """Return (has_input, has_output) — whether matching devices exist by our regex."""
    has_input  = find_input_index_by_regex(RECORD_REGEX)   is not None
    has_output = find_output_index_by_regex(PLAYBACK_REGEX) is not None
    return has_input, has_output

def check_voicemeeter_installed():
    """Return (has_input, has_audio_output, has_comm_output) — presence of Voicemeeter devices."""
    has_input = find_input_index_by_regex(VOICEMEETER_RECORD_REGEX) is not None
    has_audio_output = find_output_index_by_regex(VOICEMEETER_PLAYBACK_AUDIO_REGEX) is not None
    has_comm_output = find_output_index_by_regex(VOICEMEETER_PLAYBACK_COMM_REGEX) is not None
    return has_input, has_audio_output, has_comm_output

# ---------------------------------
# VAD (Voice Activity Detection)
# ---------------------------------
class VADProcessor:
    """Processor for detecting voice activity"""
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 1):
        if not VAD_AVAILABLE:
            raise RuntimeError("webrtcvad is not installed. pip install webrtcvad")
        
        # VAD works only with specific sample rates
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"VAD supports only 8k, 16k, 32k, 48k Hz. Got: {sample_rate}")
        
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3, higher = more aggressive
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # WebRTC VAD requires frames multiple of 10ms
        self.frame_length = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Buffer for accumulating audio
        self.buffer = np.array([], dtype=np.float32)
        
        # Filtering settings
        self.min_speech_frames = 3   # min consecutive speech frames
        self.min_silence_frames = 10 # min consecutive silence frames to end
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        
        logger.info(f"VAD initialized: sr={sample_rate}, aggressiveness={aggressiveness}")

    def _convert_to_int16(self, audio: np.ndarray) -> bytes:
        """Convert float32 -> int16 for VAD"""
        audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process audio chunk via VAD
        Returns: (speech_detected, accumulated_audio_or_None)
        """
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Process frames of frame_length
        speech_detected = False
        processed_frames = 0
        
        while len(self.buffer) >= self.frame_length:
            frame = self.buffer[:self.frame_length]
            self.buffer = self.buffer[self.frame_length:]
            processed_frames += 1
            
            # VAD analysis
            try:
                frame_bytes = self._convert_to_int16(frame)
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                logger.debug(f"VAD error: {e}")
                is_speech = True  # treat as speech on error
            
            if is_speech:
                self.speech_frame_count += 1
                self.silence_frame_count = 0
                speech_detected = True
            else:
                self.silence_frame_count += 1
                self.speech_frame_count = max(0, self.speech_frame_count - 1)
        
        # State logic
        if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
            self.is_speaking = True
            logger.debug("Speech started")
        elif self.is_speaking and self.silence_frame_count >= self.min_silence_frames:
            self.is_speaking = False
            logger.debug("Speech ended")
            # Return accumulated buffer
            if len(self.buffer) > 0:
                result_audio = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                return True, result_audio
        
        return speech_detected, None

    def has_buffered_audio(self) -> bool:
        """Whether there is accumulated audio in the buffer"""
        return len(self.buffer) > 0

    def get_buffered_audio(self) -> Optional[np.ndarray]:
        """Get and clear the buffer"""
        if len(self.buffer) > 0:
            result = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            return result
        return None

# ---------------------------------
# Input device selection utilities
# ---------------------------------
def find_input_index_by_regex(pattern: str) -> Optional[int]:
    rx = re.compile(pattern, re.IGNORECASE)
    matches = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and rx.search(d.get("name") or ""):
            matches.append(i)
    return min(matches) if matches else None

def find_output_index_by_regex(pattern: str) -> Optional[int]:
    rx = re.compile(pattern, re.IGNORECASE)
    matches = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0 and rx.search(d.get("name") or ""):
            matches.append(i)
    return min(matches) if matches else None

def list_input_devices():
    devs = sd.query_devices()
    print("Available input devices:")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(f"[{i:>2}] {d['name']}  (in:{d['max_input_channels']}, out:{d['max_output_channels']})")

def find_device_index(substr_or_index: Optional[str]) -> Optional[int]:
    if substr_or_index is None:
        return None
    s = str(substr_or_index).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    sub = s.lower()
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0 and sub in d["name"].lower():
            return i
    return None

# ---------------------------------
# Time formatting for segments
# ---------------------------------
def _fmt_ts(sec: Optional[float]) -> str:
    if sec is None:
        return "00:00:00"
    ms = int(round(sec * 1000))
    s, ms = divmod(ms, 1000)
    m, s  = divmod(s, 60)
    h, m  = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _fmt_srt_ts(sec: float) -> str:
    """Format time for SRT subtitles"""
    ms = int(round(sec * 1000))
    s, ms = divmod(ms, 1000)
    m, s  = divmod(s, 60)
    h, m  = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ---------------------------------
# Text analyzer for sentence starts
# ---------------------------------
class SentenceAnalyzer:
    """Analyzes text to detect beginnings of new sentences"""
    
    def __init__(self):
        # Pattern for sentence start:
        # Uppercase letter after space or at the start of the string
        self.sentence_start_pattern = re.compile(r'^[A-ZА-ЯЁ]|(?<=\s)[A-ZА-ЯЁ]')
        
        # Abbreviations (not sentence endings)
        self.abbreviations = {
            'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'vs.', 'etc.', 'i.e.', 'e.g.',
            'г.', 'ул.', 'пр.', 'т.е.', 'т.д.', 'т.п.', 'и.т.д.', 'и.т.п.'
        }
    
    def starts_with_sentence(self, text: str) -> bool:
        """Check whether text starts a new sentence"""
        text = text.strip()
        if not text:
            return False
        
        # Starts with uppercase letter?
        if re.match(r'^[A-ZА-ЯЁ]', text):
            return True
        
        # Starts with a digit (can be a sentence start)
        if re.match(r'^\d', text):
            return True
            
        return False
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for analysis"""
        return re.sub(r'\s+', ' ', text.strip())

# ---------------------------------
# Console output manager with block correction
# ---------------------------------
class ConsoleOutputManager:
    def __init__(self):
        self.accumulated_content = ""   # Text drawn in console (with ANSI)
        self.visible_text = ""          # Same without ANSI (for width/line calculations)
        self.current_blocks_text = ""   # Accumulated "draft" text (unformatted)
        self.sentence_analyzer = SentenceAnalyzer()
        self.first_output = True
        self.has_blocks = False
        self.indent_size = 11

        try:
            self.terminal_width = shutil.get_terminal_size().columns
        except Exception:
            self.terminal_width = 80

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes for correct length/line counting."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def clear_accumulated_blocks(self, reset_state: bool = True):
        """
        Clears previously drawn italic blocks in the console.

        reset_state=True  — also clears internal state
                            (accumulated_content / visible_text / current_blocks_text / has_blocks).
        reset_state=False — only re-draws the console, keeps state for further growth.
        """
        if not self.accumulated_content:
            return

        # Count how many lines were printed (without ANSI), considering wrapping.
        lines_used = len(self.visible_text.split('\n')) if '\n' in self.visible_text else 1
        if lines_used == 1:
            # Estimate number of lines by terminal width
            lines_used = max(1, (len(self.visible_text) + self.terminal_width - 1) // self.terminal_width)

        # Return to the beginning of the current line
        print('\r', end='')

        # Clear current line
        current_line_pos = len(self.visible_text) % self.terminal_width
        if current_line_pos == 0 and len(self.visible_text) > 0:
            current_line_pos = self.terminal_width
        print(' ' * current_line_pos, end='')
        print('\r', end='')

        # If more lines above — clear them
        if lines_used > 1:
            for _ in range(lines_used - 1):
                if IS_WIN:
                    # Simple cross-platform variant
                    print('\033[1A', end='')                 # up one line
                    print('\r' + ' ' * self.terminal_width + '\r', end='')  # clear
                else:
                    print('\033[1A', end='')  # up one line
                    print('\033[2K', end='')  # clear entire line
                    print('\r', end='')       # to line start

        if reset_state:
            self.accumulated_content = ""
            self.visible_text = ""
            self.current_blocks_text = ""
            self.has_blocks = False

    def print_block(self, text: str):
        """Render/re-render a temporary (draft) italic block, accumulating them."""
        text = text.strip()
        if not text:
            return

        # Accumulate draft text
        if self.current_blocks_text:
            self.current_blocks_text += " " + text
        else:
            self.current_blocks_text = text

        # Re-render without resetting state
        self.clear_accumulated_blocks(reset_state=False)

        # Format and print in italics
        formatted_text = self._format_with_indent(self.current_blocks_text, self.indent_size, self.indent_size)
        lines = formatted_text.split('\n')
        formatted_lines = [f"\033[3m{line}\033[0m" for line in lines]  # ANSI italics
        self.accumulated_content = '\n'.join(formatted_lines)
        self.visible_text = self._strip_ansi(self.accumulated_content)

        print(self.accumulated_content, end='', flush=True)
        self.has_blocks = True

    def _format_with_indent(
        self,
        text: str,
        first_line_indent: int = 11,
        subsequent_indent: int = 11,
        first_line_prefix: str = ""
    ) -> str:
        """
        Format text by lines considering:
          - visible terminal width,
          - indents for first/subsequent lines,
          - prefix for the first line (e.g., timestamp).
        """
        if not text:
            return text

        words = text.split()
        if not words:
            return text

        # Visible length of prefix (without ANSI)
        prefix_visible_len = len(self._strip_ansi(first_line_prefix)) if first_line_prefix else 0

        lines = []
        # First line: prefix + indent
        current_line = (first_line_prefix or "") + (" " * first_line_indent)
        current_length = prefix_visible_len + first_line_indent  # count visible chars only

        for word in words:
            word_length = len(word)
            has_content = (len(current_line.lstrip()) > 0) and (len(self._strip_ansi(current_line)) > 0)
            extra = 1 if has_content else 0  # space between words if not line start

            # Need wrap?
            if current_length + extra + word_length > self.terminal_width:
                # Finish current line
                lines.append(current_line)

                # New line with subsequent indent (without prefix)
                current_line = (" " * subsequent_indent) + word
                current_length = subsequent_indent + word_length
            else:
                if has_content:
                    current_line += " "
                    current_length += 1
                current_line += word
                current_length += word_length

        if current_line.strip():
            lines.append(current_line)

        return "\n".join(lines)


    def print_final_text(self, text: str, timestamp: str = None, is_new_sentence: bool = False):
        """
        Final text replaces temporary blocks.
        If is_new_sentence=True and timestamp is provided — print it as a prefix of the first line
        with proper wrapping and indents.
        """
        if self.has_blocks:
            self.clear_accumulated_blocks(reset_state=True)
            time.sleep(0.01)

        text = (text or "").strip()
        if not text:
            return

        if (is_new_sentence and timestamp) or self.first_output:
            # Prefix: "[00:00:12] " (don't add brackets twice if already present)
            ts = timestamp or ""
            prefix = f"[{ts}]" if ts and not ts.startswith("[") else (ts + " " if ts else "")

            # Format whole paragraph at once considering prefix
            formatted = self._format_with_indent(
                text,
                first_line_indent=0,                 # first line goes right after prefix
                subsequent_indent=self.indent_size,  # subsequent lines with indent
                first_line_prefix=prefix
            )
            print(formatted)
        else:
            # No timestamp — regular formatting with indents
            formatted = self._format_with_indent(
                text,
                first_line_indent=self.indent_size,
                subsequent_indent=self.indent_size
            )
            print(formatted)

        self.first_output = False

# ---------------------------------
# Timestamp manager for live mode
# ---------------------------------
class LiveTimestampManager:
    """Manages timestamps for live mode"""
    def __init__(self, mode: str = "local"):
        # mode: "local" → local current time; "relative" → since session start
        self.mode = mode
        self.session_start_time = time.time()
        self.sentence_analyzer = SentenceAnalyzer()

    def should_add_timestamp(self, text: str) -> bool:
        return self.sentence_analyzer.starts_with_sentence(text)

    def get_timestamp_str(self, current_time: Optional[float] = None) -> str:
        if self.mode == "relative":
            if current_time is None:
                current_time = time.time()
            elapsed = current_time - self.session_start_time
            return _fmt_ts(elapsed)  # 00:00:12
        else:
            # local current time
            return datetime.now().strftime("%H:%M:%S")

# ---------------------------------
# Timestamp manager for file mode  
# ---------------------------------
class FileTimestampManager:
    """Manages timestamps for file mode"""
    
    def __init__(self):
        self.sentence_analyzer = SentenceAnalyzer()
        
    def should_add_timestamp(self, text: str) -> bool:
        """Decide whether to add a timestamp to the text"""
        return self.sentence_analyzer.starts_with_sentence(text)

# ---------------------------------
# Subtitle generator
# ---------------------------------
class SubtitleGenerator:
    """Generates subtitles from transcription segments"""
    def __init__(self, min_duration: float = 1.0, max_duration: float = 10.0,
                 max_duration_short: float = 5.0, short_text_threshold: int = 50,
                 max_chars: int = 200):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_duration_short = max_duration_short
        self.short_text_threshold = short_text_threshold
        self.max_chars = max_chars
        
    def _is_split_point(self, text: str, pos: int) -> bool:
        """Check whether text can be split at the given position"""
        if pos >= len(text):
            return False
        
        # End of sentence
        if pos > 0 and text[pos-1] in '.!?':
            return True
        
        # Comma, colon, dash
        if pos > 0 and text[pos-1] in ',;:–—-':
            # Ensure a space follows
            if pos < len(text) and text[pos] == ' ':
                return True
        
        return False
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split long text into parts"""
        if len(text) <= self.max_chars:
            return [text]
        
        parts = []
        current_start = 0
        
        while current_start < len(text):
            # Find split point
            end_pos = min(current_start + self.max_chars, len(text))
            
            # Last chunk?
            if end_pos >= len(text):
                parts.append(text[current_start:].strip())
                break
            
            # Find nearest split point
            best_split = end_pos
            for pos in range(end_pos, current_start + 50, -1):  # Minimum 50 chars
                if self._is_split_point(text, pos):
                    best_split = pos
                    break
            
            # If not found, split by space
            if best_split == end_pos:
                for pos in range(end_pos, current_start + 50, -1):
                    if pos < len(text) and text[pos] == ' ':
                        best_split = pos
                        break
            
            parts.append(text[current_start:best_split].strip())
            current_start = best_split
            
            # Skip spaces and punctuation
            while current_start < len(text) and text[current_start] in ' ,:;–—-':
                current_start += 1
        
        return parts

    def generate_subtitles(self, segments: List[Dict]) -> List[Dict]:
        """Generate subtitles from segments with optimal splitting"""
        if not segments:
            return []

        subtitles = []
        subtitle_id = 1
        last_end_time = 0.0
        
        for segment in segments:
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            start_time = segment.get('start', 0.0)
            end_time = segment.get('end', start_time + self.min_duration)
            
            # If gap between subtitles is less than 1 sec, group them
            if start_time - last_end_time < 1.0 and subtitles:
                # Append to previous subtitle
                last_subtitle = subtitles[-1]
                last_subtitle['text'] += ' ' + text
                last_subtitle['end'] = end_time
                
                # If it became too long — split it
                if len(last_subtitle['text']) > self.max_chars:
                    # Split the last subtitle
                    parts = self._split_long_text(last_subtitle['text'])
                    duration = last_subtitle['end'] - last_subtitle['start']
                    
                    # Replace last subtitle with the first part
                    last_subtitle['text'] = parts[0]
                    part_duration = max(self.min_duration, duration / len(parts)) if len(parts) > 0 else self.min_duration
                    last_subtitle['end'] = last_subtitle['start'] + part_duration
                    
                    # Add remaining parts
                    for i, part in enumerate(parts[1:], 1):
                        subtitles.append({
                            'id': subtitle_id,
                            'start': last_subtitle['start'] + part_duration * i,
                            'end': min(last_subtitle['start'] + part_duration * (i + 1), end_time),
                            'text': part
                        })
                        subtitle_id += 1
                
                last_end_time = subtitles[-1]['end']
                continue
                
            # Split long text if needed
            text_parts = self._split_long_text(text)
            
            for i, part in enumerate(text_parts):
                part_start = start_time if i == 0 else last_end_time
                
                # Duration for this part
                is_short = len(part) < self.short_text_threshold
                max_dur = self.max_duration_short if is_short else self.max_duration
                 
                # Compute duration
                if i < len(text_parts) - 1:
                    # Not last part — fixed duration
                    duration = self.min_duration
                else:
                    # Last part — use remaining time
                    duration = end_time - part_start
                    duration = max(self.min_duration, min(duration, max_dur))

                part_end = part_start + duration

                # Ensure subtitle does not exceed segment
                if part_end > end_time:
                    part_end = end_time
                
                # Ensure minimal duration
                if part_end - part_start < self.min_duration:
                    part_end = part_start + self.min_duration
                
                subtitles.append({
                    'id': subtitle_id,
                    'start': part_start,
                    'end': part_end,
                    'text': part
                })
                subtitle_id += 1
                last_end_time = part_end
        
        # Post-process: remove overlaps and adjust timings
        for i in range(1, len(subtitles)):
            if subtitles[i]['start'] < subtitles[i-1]['end']:
                # Overlap
                gap = 0.1  # Minimal gap between subtitles
                subtitles[i]['start'] = subtitles[i-1]['end'] + gap
                
                # Ensure minimal duration
                if subtitles[i]['end'] - subtitles[i]['start'] < self.min_duration:
                    subtitles[i]['end'] = subtitles[i]['start'] + self.min_duration
        
        # Final check: remove empty subtitles
        subtitles = [s for s in subtitles if s['text'].strip()]
        
        # Re-number
        for i, sub in enumerate(subtitles, 1):
            sub['id'] = i                

        return subtitles
    
    def save_srt(self, subtitles: List[Dict], filename: str):
        """Save subtitles in SRT format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub['id']}\n")
                f.write(f"{_fmt_srt_ts(sub['start'])} --> {_fmt_srt_ts(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")

# ---------------------------------
# Progress for transcription
# ---------------------------------
def get_media_duration(file_path: str) -> Optional[float]:
    """Get media duration via ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except Exception as e:
        logger.debug(f"Could not determine file duration: {e}")
    return None

# ---------------------------------
# File transcription
# ---------------------------------
def transcribe_file(model, file_path: str, args) -> Dict:
    """Transcribe a file with progress logging"""
    logger.info(f"Transcribing file: {file_path}")
    
    duration = get_media_duration(file_path)
    if duration:
        logger.info(f"File duration: {duration:.1f} sec")
    
    task = "translate" if args.translate else "transcribe"
    
    try:
        print("Transcription started...", flush=True)
        
        result = model.transcribe(
            file_path,
            task=task,
            language=None if args.language == "auto" else args.language,
            fp16=args._use_fp16,
            temperature=0.0,
            no_speech_threshold=args.no_speech_threshold,
            condition_on_previous_text=True,
            word_timestamps=True,
            verbose=False
        )

        if not result:
            raise Exception("Whisper returned an empty result")
        
        if "text" not in result:
            raise Exception("Whisper could not recognize text in the file")        

        print("\nTranscription finished!")
        return result
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        raise

# ---------------------------------
# Audio buffer for live mode
# ---------------------------------
class AudioBuffer:
    """Buffer to accumulate and manage audio data"""
    
    def __init__(self, max_duration_sec: float = 30.0, sample_rate: int = 16000):
        self.max_samples = int(max_duration_sec * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)
        
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio to the buffer"""
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Limit buffer size
        if len(self.buffer) > self.max_samples:
            excess = len(self.buffer) - self.max_samples
            self.buffer = self.buffer[excess:]
    
    def get_audio(self) -> np.ndarray:
        """Get the whole accumulated audio"""
        return self.buffer.copy()
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = np.array([], dtype=np.float32)
    
    def duration_seconds(self) -> float:
        """Get buffer duration in seconds"""
        return len(self.buffer) / self.sample_rate

# -------------
# Main
# -------------
def main():
    ap = argparse.ArgumentParser(description="Whisper offline: live (device/cable), file (--file), or subtitles (--srt)")
    ap.add_argument("--file", type=str, default=None,
                    help="Path to audio/video file to transcribe")
    ap.add_argument("--srt", type=str, default=None,
                    help="Path to audio/video file to create subtitles (.srt)")
    ap.add_argument("--voicemeeter", action="store_true",
                    help="Auto: configures Voicemeeter (buds3.xml/desktop.xml depending on headphones), "
                         "sets Voicemeeter Input (Audio) and AUX Input (Communications). Ignores --device.")
    ap.add_argument("--cable", action="store_true",
                    help="Auto: system playback='CABLE Input'. Input device remains unchanged. Ignores --device.")
    ap.add_argument("--no-control", action="store_true",
                    help="If using --cable, decides whether to show the audio settings panel. Useful for redirecting to another output device.")
    ap.add_argument("--device", type=str, default=None,
                    help="Name-substring or index of input device (when not using --cable or --voicemeeter). Example: 'CABLE Output' or 2")
    ap.add_argument("--model", type=str, default="base", help="Whisper: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", type=str, default="en", help="Language code (ru/en/...), 'auto' = auto-detect")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate (for live; files are resampled via ffmpeg)")
    ap.add_argument("--block-sec", type=float, default=1, help="Live: block length in seconds")
    ap.add_argument("--channels", type=int, default=0, help="Live: 0=auto (up to 2), otherwise explicit number of channels")
    ap.add_argument("--translate", action="store_true", help="Translate to English (task=translate)")
    ap.add_argument("--no-speech-threshold", type=float, default=0.6, 
                    help="Whisper no_speech threshold (0.0-1.0). Higher = more aggressive noise filtering. Default: 0.6")
    ap.add_argument("--vad", action="store_true", help="Enable Voice Activity Detection (silence filtering)")
    ap.add_argument("--vad-aggressiveness", type=int, default=1, choices=[0, 1, 2, 3],
                    help="VAD aggressiveness (0=less aggressive, 3=more aggressive)")
    ap.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    
    args = ap.parse_args()

    if args.list_devices and not args.file and not args.srt:
        list_input_devices()
        return

    # Mutually exclusive modes
    if args.cable and args.voicemeeter:
        logger.error("Cannot use --cable and --voicemeeter simultaneously")
        sys.exit(1)

    # VAD check
    if args.vad and not VAD_AVAILABLE:
        logger.error("VAD requested but webrtcvad is not installed: pip install webrtcvad")
        sys.exit(1)

    # Prepare whisper
    logger.info(f"Loading Whisper model: {args.model}")

    try:
        target_dev = "mps" if (IS_MAC and MPS_AVAILABLE) else "cpu"
        logger.info(f"Selected device target: {target_dev}")
      
        # Load on CPU first to dodge SparseMPS during state dict load
        model = whisper.load_model(args.model, device="cpu")
        args._device = "cpu"
        args._use_fp16 = False # keep fp32 for stability

        if target_dev == "mps":
            try:
                # Move tensors to MPS *after* loading
                model.to("mps")
                logger.info("Moved model to MPS (fp32).")
                args._device = "mps"
                args._use_fp16 = False # stay fp32 on MPS
            except Exception as move_e:
                logger.warning(f"MPS move failed ({move_e}); staying on CPU.")

        logger.info(f"Model {args.model} ready on {args._device}")

    except Exception as e:
        if "SparseMPS" in str(e) or "_sparse_coo_tensor" in str(e):
            logger.warning("SparseMPS op missing; forcing CPU.")
            model = whisper.load_model(args.model, device="cpu")
            args._device = "cpu"
            args._use_fp1ó = False
        else:
            logger.error(f"Whisper model load error: {e}")
            sys.exit(1)

    # --------------------
    # SUBTITLES MODE
    # --------------------
    if args.srt:
        logger.info("Starting in subtitles creation mode")
        if not os.path.isfile(args.srt):
            logger.error(f"File not found: {args.srt}")
            sys.exit(1)
            
        try:
            result = transcribe_file(model, args.srt, args)
            
            if "segments" not in result:
                logger.error("Could not get segments for subtitles creation")
                sys.exit(1)
                
            # Create subtitles from adjusted text in segments
            sub_gen = SubtitleGenerator(
                min_duration=1.0,          # At least 1 second
                max_duration=10.0,         # Up to 10 seconds for long phrases
                max_duration_short=5.0,    # Up to 5 seconds for short phrases
                short_text_threshold=50,   # Short phrase < 50 chars
                max_chars=200              # Max characters per subtitle
            )
            
            subtitles = sub_gen.generate_subtitles(result["segments"])
            
            # Determine output filename
            base_name = os.path.splitext(args.srt)[0]
            srt_file = f"{base_name}.srt"
            
            sub_gen.save_srt(subtitles, srt_file)
            
            logger.info(f"Subtitles created: {len(subtitles)}")
            logger.info(f"Subtitle file saved: {srt_file}")
            
        except Exception as e:
            logger.error(f"Error while creating subtitles: {e}")
            sys.exit(1)
        return

    # --------------------
    # FILE TRANSCRIPTION MODE
    # --------------------
    if args.file:
        logger.info("Starting in file transcription mode")
        if not os.path.isfile(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        
        try:
            result = transcribe_file(model, args.file, args)
            
            # Initialize timestamp and output managers for file mode
            timestamp_manager = FileTimestampManager()
            output_manager = ConsoleOutputManager()
            
            if "segments" in result:
                # Print adjusted text with timestamps based on sentence starts
                for seg in result["segments"]:
                    start_time = seg.get("start", 0.0)
                    text = (seg.get("text") or "").strip()
                    if text:
                        is_new_sentence = timestamp_manager.should_add_timestamp(text)
                        timestamp_str = _fmt_ts(start_time) if is_new_sentence else None
                        output_manager.print_final_text(text, timestamp_str, is_new_sentence)
                print()  # final newline
            else:
                # If there are no segments, print the whole text with initial timestamp
                text_result = (result.get("text") or "").strip()
                if text_result:
                    output_manager.print_final_text(text_result, "00:00:00", True)
                    print()
                else:
                    logger.warning("No text recognized in the file")
                    
        except Exception as e:
            logger.error(f"Error during file transcription: {e}")
            sys.exit(1)
        return

    # --------------------
    # LIVE MODE (cable/device)
    # --------------------
    logger.info("Starting in live mode")
    # Check VB-CABLE / Voicemeeter before starting
    if args.voicemeeter:
        has_input, has_audio_out, has_comm_out = check_voicemeeter_installed()
        if not has_input or not has_audio_out or not has_comm_out:
            logger.error("Voicemeeter is not fully installed!")
            logger.error(f"   Voicemeeter Out B1: {'✅' if has_input else '❌'}")
            logger.error(f"   Voicemeeter Input: {'✅' if has_audio_out else '❌'}")
            logger.error(f"   Voicemeeter AUX Input: {'✅' if has_comm_out else '❌'}")
            logger.error("   Install Voicemeeter: https://vb-audio.com/Voicemeeter/")
            # If only input is missing but outputs exist, Voicemeeter may not be running
            if has_audio_out or has_comm_out:
                logger.info("   Try starting Voicemeeter manually before running this script")
            sys.exit(1)
    elif args.cable:
        has_input, has_output = check_vb_cable_installed()
        if not has_input or not has_output:
            logger.error("VB-CABLE is not fully installed!")
            logger.error(f"   CABLE Input: {'✅' if has_input else '❌'}")
            logger.error(f"   CABLE Output: {'✅' if has_output else '❌'}")
            logger.error("   Install VB-CABLE: https://vb-audio.com/Cable/")
            sys.exit(1)

    # VAD sample rate check
    if args.vad and args.sr not in [8000, 16000, 32000, 48000]:
        logger.error(f"VAD supports only 8k, 16k, 32k, 48k Hz. Provided: {args.sr}")
        logger.info("It is recommended to use --sr 16000 with VAD")
        sys.exit(1)

    block_frames = int(args.sr * args.block_sec)

    # Determine device for sounddevice
    dev_index = None
    device_name_for_log = "(default input)"
    
    # In --cable mode try to find CABLE Output for recording
    if args.voicemeeter:
        voicemeeter_input_index = find_input_index_by_regex(VOICEMEETER_RECORD_REGEX)
        if voicemeeter_input_index is not None:
            dev_index = voicemeeter_input_index
            logger.info(f"Found Voicemeeter Out B1 for recording: index {dev_index}")
        else:
            logger.warning("Voicemeeter Out B1 for recording not found, using default device")
    elif args.cable:
        cable_input_index = find_input_index_by_regex(RECORD_REGEX)
        if cable_input_index is not None:
            dev_index = cable_input_index
            logger.info(f"Found CABLE Output for recording: index {dev_index}")
        else:
            logger.warning("CABLE Output for recording not found, using default device")
            logger.info("Ensure VB-CABLE is installed correctly")
    else:
        # Normal mode — use user-specified device or default
        dev_index = find_device_index(args.device) if args.device else None
        if args.device and dev_index is None:
            logger.error(f"Input device '{args.device}' not found. Run with --list-devices.")
            sys.exit(1)

    # Check channels/description
    try:
        if dev_index is not None:
            logger.info(f"Using device index: {dev_index}")
            try:
                dev_info = sd.query_devices(dev_index)
            except sd.DeviceUnavailableError:
                logger.error(f"Device with index {dev_index} is unavailable")
                sys.exit(1)
        else:
            try:
                # Get default input device only
                default_input = sd.default.device['input'] if hasattr(sd.default.device, '__getitem__') else None
                if default_input is None:
                    # Fallback: find first available input device
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device.get('max_input_channels', 0) > 0:
                            default_input = i
                            break
                    if default_input is None:
                        raise Exception("No input audio device found")
                
                in_idx = default_input
                dev_info = sd.query_devices(in_idx)
                logger.info(f"Using default input device: index {in_idx}")
            except Exception as e:
                # Alternative: try any input device
                logger.warning(f"Problem with default device: {e}")
                logger.info("Searching for available input devices...")
                
                devices = sd.query_devices()
                input_devices = [(i, d) for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]
                
                if not input_devices:
                    logger.error("No input audio device found")
                    logger.info("Check your microphone or audio interface connection")
                    sys.exit(1)
                
                # Use the first available input device
                in_idx, dev_info = input_devices[0]
                logger.info(f"Selected first available input device: index {in_idx}, '{dev_info['name']}'")
         
        max_in = dev_info.get("max_input_channels", 0)
        if max_in <= 0:
            logger.error(f"Device '{dev_info.get('name', 'Unknown')}' does not support audio capture")
            logger.info("Select another device using --device or view the list with --list-devices")
            sys.exit(1)
        
        channels = args.channels if args.channels > 0 else min(2, max_in)
        if channels > max_in:
            logger.error(f"Requested channels: {channels}, device supports only {max_in}.")
            sys.exit(1)
        
        device_name_for_log = dev_info["name"] if dev_index is not None else f"(default) {dev_info['name']}"
        if args.voicemeeter and dev_index == voicemeeter_input_index:
            logger.info(f"Input device (Voicemeeter Out B1): {device_name_for_log}")
        elif args.cable and dev_index == cable_input_index:
            logger.info(f"Input device (CABLE Output): {device_name_for_log}")
        else:
            logger.info(f"Input device: {device_name_for_log}")

        # Ensure dev_index is set in auto-detection case
        if dev_index is None:
            dev_index = in_idx
        
    except Exception as e:
        logger.error(f"Critical error during audio initialization: {e}")
        logger.info("Try:")
        logger.info("1. Run with --list-devices to see available devices")
        logger.info("2. Set the device explicitly: --device <index_or_name>")
        logger.info("3. Check microphone connection")
        sys.exit(1)

    # Initialize VAD for live mode
    vad_processor = None
    if args.vad:
        try:
            vad_processor = VADProcessor(sample_rate=args.sr, aggressiveness=args.vad_aggressiveness)
            logger.info("VAD enabled for live mode")
        except Exception as e:
            logger.error(f"Could not initialize VAD: {e}")
            sys.exit(1)

    # Initialize managers for live mode
    timestamp_manager = LiveTimestampManager(mode="local")
    output_manager = ConsoleOutputManager()
    audio_buffer = AudioBuffer(max_duration_sec=30.0, sample_rate=args.sr)
    
    def run_capture_loop():
        task = "translate" if args.translate else "transcribe"
        logger.info(f"   Input device: {device_name_for_log} | channels: {channels} | sr: {args.sr}")
        if vad_processor:
            logger.info(f"   VAD: enabled (aggressiveness={args.vad_aggressiveness})")
        
        with sd.InputStream(device=dev_index,
                            channels=channels,
                            samplerate=args.sr,
                            blocksize=block_frames,
                            dtype="float32") as stream:
            
            logger.info("Listening...")
            
            while True:
                try:
                    audio_block, status = stream.read(block_frames)
                    if status:
                        logger.debug(f"[audio status: {status}]")
                    
                    # Convert to mono
                    mono = audio_block.mean(axis=1) if channels > 1 else audio_block.reshape(-1)
                    audio_np = np.asarray(mono, dtype=np.float32)
                    
                    current_time = time.time()
                    
                    # Add audio to accumulation buffer
                    audio_buffer.add_audio(audio_np)
                    
                    # Process via VAD (if enabled)
                    if vad_processor:
                        speech_detected, processed_audio = vad_processor.process_audio(audio_np)

                        if vad_processor.is_speaking and audio_buffer.duration_seconds() > 0.5:
                            try:
                                preview_audio = audio_buffer.get_audio()
                                preview_trans = model.transcribe(preview_audio, ...)
                                preview_text = (preview_trans.get("text") or "").strip()
                                if preview_text:
                                    output_manager.print_block(preview_text)
                            except Exception as e:
                                logger.debug(f"Preview error: {e}")
                        
                        if processed_audio is not None:
                            # VAD detected end of speech — process accumulated audio
                            logger.debug(f"VAD: processing segment of {len(processed_audio)/args.sr:.2f}s")
                            
                            try:
                                # Get full audio for correction
                                buffered_audio = audio_buffer.get_audio()
                                
                                # Quick transcription for immediate display
                                block_trans = model.transcribe(
                                    processed_audio,
                                    task=task,
                                    language=None if args.language == "auto" else args.language,
                                    fp16=args._use_fp16,
                                    temperature=0.0,
                                    no_speech_threshold=args.no_speech_threshold,
                                    condition_on_previous_text=False,
                                    word_timestamps=False
                                )
                                
                                block_text = (block_trans.get("text") or "").strip()
                                if block_text:
                                    # Show block immediately
                                    output_manager.print_block(block_text)

                                # Full transcription of accumulated audio (for correction)
                                if audio_buffer.duration_seconds() > 1.0:  # at least 1 sec for correction
                                    full_trans = model.transcribe(
                                        buffered_audio,
                                        task=task,
                                        language=None if args.language == "auto" else args.language,
                                        fp16=args._use_fp16,
                                        temperature=0.0,
                                        no_speech_threshold=args.no_speech_threshold,
                                        condition_on_previous_text=True,
                                        word_timestamps=False
                                    )
                                    
                                    full_text = (full_trans.get("text") or "").strip()
                                    if full_text and full_text != block_text:
                                        # Replace block with corrected text
                                        is_new_sentence = timestamp_manager.should_add_timestamp(full_text)
                                        timestamp_str = timestamp_manager.get_timestamp_str(current_time) if is_new_sentence else None
                                        output_manager.print_final_text(full_text, timestamp_str, is_new_sentence)
                                        
                                        # Clear buffer after correction
                                        audio_buffer.clear()
                                        
                            except Exception as e:
                                logger.error(f"Transcription error for VAD segment: {e}")
                        
                        elif not speech_detected:
                            # Occasionally log silence indicator
                            logger.debug("Silence...")
                            
                    else:
                        # No VAD — process each block
                        try:
                            # Quick transcription
                            block_trans = model.transcribe(
                                audio_np,
                                task=task,
                                language=None if args.language == "auto" else args.language,
                                fp16=args._use_fp16,
                                temperature=0.0,
                                no_speech_threshold=args.no_speech_threshold,
                                condition_on_previous_text=False,
                                word_timestamps=False
                            )
                            
                            block_text = (block_trans.get("text") or "").strip()
                            if block_text:
                                # Show block immediately
                                output_manager.print_block(block_text)
                            
                            # Full transcription of accumulated audio (every few blocks)
                            if audio_buffer.duration_seconds() > 4.0:  # every 4 seconds do correction
                                buffered_audio = audio_buffer.get_audio()
                                full_trans = model.transcribe(
                                    buffered_audio,
                                    task=task,
                                    language=None if args.language == "auto" else args.language,
                                    fp16=args._use_fp16,
                                    temperature=0.0,
                                    no_speech_threshold=args.no_speech_threshold,
                                    condition_on_previous_text=True,
                                    word_timestamps=False
                                )
                                
                                full_text = (full_trans.get("text") or "").strip()
                                if full_text and full_text != block_text:
                                    # Replace with corrected text
                                    is_new_sentence = timestamp_manager.should_add_timestamp(full_text)
                                    timestamp_str = timestamp_manager.get_timestamp_str(current_time) if is_new_sentence else None
                                    output_manager.print_final_text(full_text, timestamp_str, is_new_sentence)
                                    
                                    # Clear buffer after correction
                                    audio_buffer.clear()
                                    
                        except Exception as e:
                            logger.error(f"Block transcription error: {e}")
                            
                except KeyboardInterrupt:
                    print()
                    logger.info("\nStopped by user.")
                    
                    # Process remaining audio in buffers
                    remaining_tasks = []
                    
                    if vad_processor and vad_processor.has_buffered_audio():
                        remaining_tasks.append(("VAD buffer", vad_processor.get_buffered_audio()))
                    
                    if audio_buffer.duration_seconds() > 0.5:
                        remaining_tasks.append(("Audio buffer", audio_buffer.get_audio()))
                    
                    if remaining_tasks:
                        logger.info("Processing remaining audio...")
                        for task_name, remaining_audio in remaining_tasks:
                            if remaining_audio is not None and len(remaining_audio) > args.sr * 0.5:
                                try:
                                    trans = model.transcribe(
                                        remaining_audio,
                                        task=task,
                                        language=None if args.language == "auto" else args.language,
                                        fp16=args._use_fp16,
                                        temperature=0.0,
                                        no_speech_threshold=args.no_speech_threshold,
                                        condition_on_previous_text=True,
                                        word_timestamps=False
                                    )
                                    text = (trans.get("text") or "").strip()
                                    if text:
                                        is_new_sentence = timestamp_manager.should_add_timestamp(text)
                                        timestamp_str = timestamp_manager.get_timestamp_str(current_time) if is_new_sentence else None
                                        output_manager.print_final_text(text, timestamp_str, is_new_sentence)
                                except Exception as e:
                                    logger.error(f"Error processing {task_name}: {e}")
                    print()  # final newline
                    break
                    
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    time.sleep(0.2)

    # Run based on mode
    try:
        if args.voicemeeter:

            with VoicemeeterSwitcher():
                try:
                    current_dev_info = sd.query_devices(dev_index)
                    device_name_for_log = f"(previously determined) {current_dev_info['name']}"
                    logger.info(f"Using input device: {device_name_for_log}")
                except Exception as e:
                    logger.warning(f"Could not refresh device info: {e}")
                    pass
                
                # Verify device availability
                try:
                    test_stream = sd.InputStream(
                        device=dev_index,
                        channels=1,
                        samplerate=args.sr,
                        blocksize=1024,
                        dtype="float32"
                    )
                    test_stream.close()
                    logger.info("Input device verified and ready")
                except Exception as e:
                    logger.error(f"Input device unavailable: {e}")
                    logger.info("Try selecting another device with --device")
                    sys.exit(1)
                
                run_capture_loop()
        elif args.cable:
            if not args.no_control:
                open_audio_settings()

            with AudioSwitcher():
                try:
                    # Use already determined device instead of querying default again
                    current_dev_info = sd.query_devices(dev_index)
                    device_name_for_log = f"(previously determined) {current_dev_info['name']}"
                    logger.info(f"Using input device: {device_name_for_log}")
                except Exception as e:
                    # Fallback to already set name
                    logger.warning(f"Could not refresh device info: {e}")
                    pass
                
                # Additional check: ensure device is available before start
                try:
                    # Test-create a stream to verify availability
                    test_stream = sd.InputStream(
                        device=dev_index,
                        channels=1,
                        samplerate=args.sr,
                        blocksize=1024,
                        dtype="float32"
                    )
                    test_stream.close()
                    logger.info("Input device verified and ready")
                except Exception as e:
                    logger.error(f"Input device unavailable: {e}")
                    logger.info("Try selecting another device with --device")
                    sys.exit(1)
                
                run_capture_loop()
        else:
            # Same check for mode without --cable
            try:
                test_stream = sd.InputStream(
                    device=dev_index,
                    channels=1,
                    samplerate=args.sr,
                    blocksize=1024,
                    dtype="float32"
                )
                test_stream.close()
                logger.info("Input device verified and ready")
            except Exception as e:
                logger.error(f"Input device unavailable: {e}")
                logger.info("Try selecting another device with --device")
                sys.exit(1)
            run_capture_loop()
            
    except KeyboardInterrupt:
        logger.info("\nProgram terminated by user")
    except sd.PortAudioError as e:
        logger.error(f"PortAudio error: {e}")
        logger.info("Try restarting the program or selecting another audio device")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("Check whisper_audio.log for diagnostics")
        sys.exit(1)

if __name__ == "__main__":
    main()
