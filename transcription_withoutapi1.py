#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Transcribe & Auto-Translate (All-in-One)
Fixed version with better streamlink handling
"""

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import threading
import webbrowser
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ffmpeg
import pysrt
from tqdm import tqdm

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# Optional (Windows-only for MPC-BE hot reload)
try:
    from pywinauto import Application as _WinApplication
except Exception:
    _WinApplication = None


# ---------------------- Utility ----------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def which(cmd):
    return shutil.which(cmd) is not None

def os_is_windows():
    return platform.system().lower().startswith("win")


# ---------------------- GPU/CPU Auto-Pick ----------------------

def pick_device_and_model(user_model=None):
    """
    - Optimized for CPU performance
    """
    # Force CPU for better stability
    device = "cpu"
    compute_type = "int8"  # Better for CPU
    
    # Use smaller models for CPU
    if user_model:
        model_name = user_model
    else:
        model_name = "small"  # Default to small for CPU
    
    return model_name, "cpu", "int8"


# ---------------------- Translation ----------------------

def choose_target_lang(detected_lang_code: str) -> str:
    """
    fr -> en, en -> fr, everything else -> en
    """
    if not detected_lang_code:
        return "en"
    code = detected_lang_code.lower()
    if code.startswith("fr"):
        return "en"
    if code.startswith("en"):
        return "fr"
    return "en"

def translate_batch(sentences, source="auto", target="en", max_retries=3):
    """
    Translate text with retry mechanism for network issues
    """
    if not sentences:
        return []

    translated = []
    for sentence in sentences:
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source, target=target)
                translated_text = translator.translate(sentence)
                translated.append(translated_text)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    log(f"Translation failed after {max_retries} attempts: {e}")
                    translated.append(sentence)  # Fallback to original text
                time.sleep(1)  # Wait before retry
    
    return translated


# ---------------------- SRT Helpers ----------------------

def format_srt_time(seconds_float):
    # Use timezone-aware method to avoid deprecation warning
    t = datetime.datetime.fromtimestamp(seconds_float, datetime.timezone.utc)
    return t.strftime("%H:%M:%S,%f")[:-3]

def append_captions_srt(caption_path, start_offset, captions, last_index):
    """
    captions: list of (start, end, text); times are relative to the chunk.
    """
    if not captions:
        return last_index
    
    try:
        with open(caption_path, "a", encoding="utf-8") as f:
            idx = last_index
            for (st, en, txt) in captions:
                abs_st = start_offset + float(st)
                abs_en = start_offset + float(en)
                idx += 1
                f.write(f"{idx}\n{format_srt_time(abs_st)} --> {format_srt_time(abs_en)}\n{txt}\n\n")
        return idx
    except Exception as e:
        log(f"Error writing to SRT file: {e}")
        return last_index

def read_existing_srt_state(caption_path):
    if not os.path.exists(caption_path):
        return 0.0, 0
    try:
        subs = pysrt.open(caption_path, encoding="utf-8")
        if subs and len(subs) > 0:
            end_ms = subs[-1].end.ordinal
            return end_ms / 1000.0, subs[-1].index
    except Exception:
        pass
    return 0.0, 0


# ---------------------- I/O: YouTube Live (streamlink) ----------------------

def streamlink_download(url, out_file="video.mp4", max_retries=5):
    if not which("streamlink"):
        raise RuntimeError("streamlink is not installed. Install with: pip install streamlink")
    
    # Clean up any existing file
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
        except:
            pass
    
    # Try different stream qualities if best doesn't work
    quality_options = ["best", "720p", "480p", "360p", "worst"]
    
    for quality in quality_options:
        for attempt in range(max_retries):
            try:
                cmd = ["streamlink", url, quality, "-o", out_file, "--force", "--retry-streams", "5", "--retry-open", "3"]
                log(f"Trying streamlink with quality {quality}: {' '.join(cmd)}")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                
                # Wait a bit to see if download starts
                time.sleep(10)
                
                # Check if file is being created
                if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    log(f"Streamlink successfully started with quality {quality}")
                    return process
                else:
                    log(f"Streamlink with quality {quality} didn't create output, trying next...")
                    try:
                        process.terminate()
                    except:
                        pass
                    break
                    
            except Exception as e:
                log(f"Streamlink failed with quality {quality}, attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    continue
                time.sleep(5)
    
    raise RuntimeError("All streamlink quality options failed")

def extract_audio_segment(input_video, output_audio, start_time=None, end_time=None):
    try:
        in_args = {}
        if start_time is not None:
            in_args["ss"] = start_time
        if end_time is not None:
            in_args["to"] = end_time
        
        (
            ffmpeg
            .input(input_video, **in_args)
            .output(output_audio, ac=1, ar=16000)  # mono 16k for speed
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        log(f"Error extracting audio: {e}")
        return False

def probe_video(path):
    try:
        info = ffmpeg.probe(path)
        video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        return float(video.get("duration", 0.0))
    except Exception:
        return 0.0


# ---------------------- Transcription Core ----------------------

def build_model(model_name=None):
    m, dev, ctype = pick_device_and_model(user_model=model_name)
    log(f"Loading whisper model: name={m}, device={dev}, compute_type={ctype}")
    model = WhisperModel(m, device=dev, compute_type=ctype)
    return model, dev

def transcribe_file(model, audio_path, vad=True):
    try:
        segs, info = model.transcribe(audio_path, vad_filter=vad, word_timestamps=False)
        caps = [(s.start, s.end, s.text.strip()) for s in segs]
        detected = getattr(info, "language", None) or ""
        return caps, detected
    except Exception as e:
        log(f"Transcription error: {e}")
        return [], ""


# ---------------------- Player (optional) ----------------------

def maybe_launch_player_and_autoreload(video_path, srt_path):
    """
    Windows-only: try to launch MPC-BE and auto-reload subs when file changes.
    """
    if not os_is_windows() or _WinApplication is None:
        return None, None
    
    # Check if video file exists
    if not os.path.exists(video_path):
        log(f"Video file not found: {video_path}")
        return None, None
        
    candidates = [
        r"C:\Program Files\MPC-BE\mpc-be64.exe",
        r"C:\Program Files (x86)\MPC-BE\mpc-be.exe",
        r"C:\Program Files\MPC-HC\mpc-hc64.exe",
        r"C:\Program Files (x86)\MPC-HC\mpc-hc.exe",
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    ]
    exe = next((p for p in candidates if os.path.exists(p)), None)
    if exe is None:
        log("Media player not found; skipping player.")
        return None, None
    
    try:
        if "vlc" in exe.lower():
            proc = subprocess.Popen([exe, video_path, "--sub-file", srt_path])
        else:
            proc = subprocess.Popen([exe, video_path, "/sub", srt_path])
            
        log(f"Launched media player: {exe}")
        time.sleep(5.0)  # Give more time for player to start
        
        return proc, None
    except Exception as e:
        log(f"Failed to launch player: {e}")
        return None, None


# ---------------------- Pipelines ----------------------

def pipeline_from_video_file(args):
    video_path = args.file
    srt_path = args.srt
    delay = args.delay
    wait_time = args.wait

    model, _ = build_model(args.model)

    player_proc, watcher_th = None, None
    if args.play:
        player_proc, watcher_th = maybe_launch_player_and_autoreload(video_path, srt_path)

    last_end, last_idx = read_existing_srt_state(srt_path)
    log(f"Resuming at t={last_end:.1f}s, index={last_idx}")

    # Wait for video file to be created and have content
    max_wait_time = 120  # Maximum time to wait for video file (seconds)
    start_time = time.time()
    
    while not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        if time.time() - start_time > max_wait_time:
            log("Timeout waiting for video file. Exiting.")
            return
            
        log("Waiting for video file to be created...")
        time.sleep(5)

    try:
        while True:
            # Check if video file exists and is accessible
            if not os.path.exists(video_path):
                log(f"Video file not found: {video_path}")
                time.sleep(wait_time)
                continue
                
            dur = probe_video(video_path)
            if dur <= last_end + 1.0:  # Wait until we have at least 1 second of new content
                time.sleep(wait_time)
                continue

            start_t = last_end
            end_t = dur
            last_end = end_t

            tmp_audio = "tmp_chunk.wav"
            if not extract_audio_segment(video_path, tmp_audio, start_t, end_t):
                time.sleep(wait_time)
                continue

            caps, detected = transcribe_file(model, tmp_audio, vad=True)

            tgt = choose_target_lang(detected) if args.auto_translate else None
            if tgt and caps:
                src = "auto"
                caps_txt = [c[2] for c in caps]
                tr_txt = translate_batch(caps_txt, source=src, target=tgt)
                caps = [(c[0], c[1], t) for c, t in zip(caps, tr_txt)]

            last_idx = append_captions_srt(srt_path, start_t, caps, last_idx)
            log(f"Appended {len(caps)} captions up to {end_t:.1f}s (detected={detected or 'n/a'}, translated_to={tgt or 'off'})")
            
            # Clean up
            try:
                if os.path.exists(tmp_audio):
                    os.remove(tmp_audio)
            except:
                pass
                
    except KeyboardInterrupt:
        log("Stopping pipeline.")
    except Exception as e:
        log(f"Error in pipeline: {e}")
    finally:
        if player_proc and player_proc.poll() is None:
            try: 
                player_proc.terminate()
            except Exception: 
                pass

def pipeline_from_url(args):
    out_file = args.file or "video.mp4"
    
    # Create SRT file if it doesn't exist
    if not os.path.exists(args.srt):
        with open(args.srt, 'w', encoding='utf-8') as f:
            f.write('')
    
    sl = None
    try:
        sl = streamlink_download(args.url, out_file=out_file)
        # Wait a bit for streamlink to start downloading
        time.sleep(10)
        pipeline_from_video_file(args)
    except Exception as e:
        log(f"Error in URL pipeline: {e}")
    finally:
        try:
            if sl and sl.poll() is None:
                sl.terminate()
        except Exception:
            pass


# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser(description="Live Transcribe & Auto-Translate (All-in-One)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Live URL (YouTube, Twitch, etc.). Uses streamlink.")
    src.add_argument("--file", help="Existing/growing video file (default: video.mp4).", default="video.mp4")

    p.add_argument("--srt", default="captions.srt", help="Path to SRT output (appended continuously).")
    p.add_argument("--delay", type=int, default=10, help="For video inputs, wait this many seconds of new content before transcribing.")
    p.add_argument("--wait", type=int, default=2, help="Polling interval in seconds.")
    p.add_argument("--model", default="tiny", help="Whisper model (e.g., tiny, small, medium, large-v3). Default: tiny")
    p.add_argument("--auto-translate", action="store_true", help="Enable auto translation: fr->en, en->fr, others->en")
    p.add_argument("--play", action="store_true", help="On Windows, try to launch media player and auto-reload captions.")

    args = p.parse_args()

    if args.url:
        if args.file is None:
            args.file = "video.mp4"
        pipeline_from_url(args)
    elif args.file:
        pipeline_from_video_file(args)
    else:
        print("Nothing to do. Provide --url or --file.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
