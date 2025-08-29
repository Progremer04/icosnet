#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Transcribe & Auto-Translate API Server
Optimized for speed and real-time performance
"""

import argparse
import asyncio
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import threading
import uuid
import webbrowser
import warnings
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ffmpeg
import pysrt
from tqdm import tqdm
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Optional (Windows-only for MPC-BE hot reload)
try:
    from pywinauto import Application as _WinApplication
except Exception:
    _WinApplication = None

# Import transcription modules
try:
    from faster_whisper import WhisperModel
    from deep_translator import GoogleTranslator
except ImportError:
    print("Please install required packages: pip install faster-whisper deep-translator")
    sys.exit(1)

# ---------------------- Configuration ----------------------

# Session management
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# Performance settings
MAX_WORKERS = 8
CHUNK_SIZE = 5  # seconds of audio per chunk
POLL_INTERVAL = 1  # seconds between checks
VAD_FILTER = True
COMPUTE_TYPE = "int8"  # Better for CPU

# Language settings
LANGUAGE_MAPPING = {
    "fr": "French",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian"
}

# ---------------------- FastAPI App Setup ----------------------

app = FastAPI(
    title="Live Transcription API Server",
    description="Optimized real-time transcription and translation API with WebSocket support",
    version="4.0.0"
)

# Enable CORS for network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)

# Thread pool for concurrent processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
process_pool = ProcessPoolExecutor(max_workers=2)  # For CPU-intensive tasks
# Initialize global variables
active_sessions: Dict[str, Any] = {}
session_threads: Dict[str, threading.Thread] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}
session_locks: Dict[str, threading.Lock] = {}
# ---------------------- Data Models ----------------------

class TranscriptionRequest(BaseModel):
    url: Optional[str] = None
    file_path: Optional[str] = None
    srt_path: str = "captions.srt"
    model: str = "small"
    auto_translate: bool = True
    target_language: Optional[str] = None
    poll_interval: int = POLL_INTERVAL
    play: bool = False
    low_latency: bool = True

class TranscriptionStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"

class TranscriptionSession(BaseModel):
    id: str
    status: TranscriptionStatus
    request: TranscriptionRequest
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    progress: float = 0.0
    message: str = ""
    detected_language: Optional[str] = None
    target_language: Optional[str] = None
    latest_captions: List[Dict] = []
    total_captions: int = 0
    session_dir: str = ""
    video_path: str = ""
    srt_path: str = ""

# ---------------------- Utility Functions ----------------------

def log(msg, session_id=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    session_prefix = f"[Session {session_id[:8]}] " if session_id else ""
    print(f"[{ts}] {session_prefix}{msg}", flush=True)

def which(cmd):
    return shutil.which(cmd) is not None

def os_is_windows():
    return platform.system().lower().startswith("win")

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Create a socket connection to get the local IP
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_session_dir(session_id):
    """Get the directory for a session"""
    return os.path.join(SESSION_DIR, session_id)

def cleanup_session(session_id):
    """Clean up session files"""
    try:
        session_dir = get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        log(f"Cleaned up session files: {session_dir}", session_id)
    except Exception as e:
        log(f"Error cleaning up session: {e}", session_id)

# ---------------------- GPU/CPU Auto-Pick ----------------------

def pick_device_and_model(user_model=None):
    """
    - Optimized for CPU performance
    """
    # Force CPU for better stability
    device = "cpu"
    compute_type = COMPUTE_TYPE  # Better for CPU
    
    # Use smaller models for CPU
    if user_model:
        model_name = user_model
    else:
        model_name = "small"  # Default to small for CPU
    
    return model_name, "cpu", compute_type

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
                time.sleep(0.5)  # Wait before retry
    
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

def streamlink_download(url, out_file="video.mp4", max_retries=5, session_id=None):
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
                log(f"Trying streamlink with quality {quality}: {' '.join(cmd)}", session_id)
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                
                # Wait a bit to see if download starts
                time.sleep(5)
                
                # Check if file is being created
                if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    log(f"Streamlink successfully started with quality {quality}", session_id)
                    return process
                else:
                    log(f"Streamlink with quality {quality} didn't create output, trying next...", session_id)
                    try:
                        process.terminate()
                    except:
                        pass
                    break
                    
            except Exception as e:
                log(f"Streamlink failed with quality {quality}, attempt {attempt+1}: {e}", session_id)
                if attempt == max_retries - 1:
                    continue
                time.sleep(3)
    
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
        time.sleep(3.0)  # Give time for player to start
        
        return proc, None
    except Exception as e:
        log(f"Failed to launch player: {e}")
        return None, None

# ---------------------- WebSocket Manager ----------------------

async def broadcast_to_websockets(session_id: str, message: dict):
    """Broadcast a message to all WebSocket connections for a session"""
    if session_id in websocket_connections:
        disconnected = []
        for ws in websocket_connections[session_id]:
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(ws)
            except Exception as e:
                log(f"WebSocket error: {e}", session_id)
        
        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections[session_id].remove(ws)

# ---------------------- Transcription Session Manager ----------------------

def run_transcription_session(session_id: str, request: TranscriptionRequest):
    """Main function to run transcription in a background thread"""
    session = active_sessions[session_id]
    
    try:
        # Update session status
        session.status = TranscriptionStatus.STARTING
        asyncio.run(broadcast_to_websockets(session_id, {
            "type": "status_update",
            "status": session.status,
            "message": "Starting transcription session"
        }))
        
        # Create session directory
        session_dir = get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Set up paths
        video_path = os.path.join(session_dir, "video.mp4")
        srt_path = os.path.join(session_dir, "captions.srt")
        
        # Update session with paths
        session.session_dir = session_dir
        session.video_path = video_path
        session.srt_path = srt_path
        
        model, _ = build_model(request.model)
        
        # Create SRT file if it doesn't exist
        if not os.path.exists(srt_path):
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write('')
        
        # Handle URL source
        sl_process = None
        if request.url:
            try:
                sl_process = streamlink_download(request.url, out_file=video_path, session_id=session_id)
                # Wait a bit for streamlink to start downloading
                time.sleep(5)
            except Exception as e:
                error_msg = f"Streamlink failed: {str(e)}"
                log(error_msg, session_id)
                session.status = TranscriptionStatus.ERROR
                session.message = error_msg
                asyncio.run(broadcast_to_websockets(session_id, {
                    "type": "error",
                    "message": error_msg
                }))
                return
        
        # Update session status
        session.status = TranscriptionStatus.RUNNING
        session.start_time = datetime.datetime.now()
        asyncio.run(broadcast_to_websockets(session_id, {
            "type": "status_update",
            "status": session.status,
            "message": "Transcription in progress",
            "start_time": session.start_time.isoformat()
        }))
        
        last_end, last_idx = read_existing_srt_state(srt_path)
        log(f"Resuming at t={last_end:.1f}s, index={last_idx}", session_id)
        
        # Wait for video file to be created and have content
        max_wait_time = 60  # Maximum time to wait for video file (seconds)
        start_time = time.time()
        
        while not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            if time.time() - start_time > max_wait_time:
                error_msg = "Timeout waiting for video file"
                log(error_msg, session_id)
                session.status = TranscriptionStatus.ERROR
                session.message = error_msg
                asyncio.run(broadcast_to_websockets(session_id, {
                    "type": "error",
                    "message": error_msg
                }))
                return
                
            log("Waiting for video file to be created...", session_id)
            time.sleep(2)
        
        # Launch player if requested
        player_proc = None
        if request.play:
            player_proc, _ = maybe_launch_player_and_autoreload(video_path, srt_path)
        
        # Main transcription loop
        while session.status == TranscriptionStatus.RUNNING:
            # Check if video file exists and is accessible
            if not os.path.exists(video_path):
                log(f"Video file not found: {video_path}", session_id)
                time.sleep(request.poll_interval)
                continue
                
            dur = probe_video(video_path)
            if dur <= last_end + 1.0:  # Wait until we have at least 1 second of new content
                time.sleep(request.poll_interval)
                continue

            # Use smaller chunks for lower latency if requested
            chunk_size = CHUNK_SIZE
            if request.low_latency:
                chunk_size = max(2, CHUNK_SIZE // 2)  # Use smaller chunks for lower latency
            
            start_t = last_end
            end_t = min(dur, start_t + chunk_size)  # Process smaller chunks
            last_end = end_t

            tmp_audio = os.path.join(session_dir, f"chunk_{uuid.uuid4().hex}.wav")
            if not extract_audio_segment(video_path, tmp_audio, start_t, end_t):
                time.sleep(request.poll_interval)
                continue

            caps, detected = transcribe_file(model, tmp_audio, vad=VAD_FILTER)
            
            # Update detected language in session
            if detected and not session.detected_language:
                session.detected_language = detected
                session.target_language = request.target_language or choose_target_lang(detected)
                asyncio.run(broadcast_to_websockets(session_id, {
                    "type": "language_detected",
                    "detected": detected,
                    "target": session.target_language,
                    "detected_name": LANGUAGE_MAPPING.get(detected.split('-')[0], detected),
                    "target_name": LANGUAGE_MAPPING.get(session.target_language, session.target_language)
                }))

            # Translate if enabled
            tgt = session.target_language if request.auto_translate else None
            if tgt and caps:
                src = "auto"
                caps_txt = [c[2] for c in caps]
                # Use thread pool for translation to avoid blocking
                future = thread_pool.submit(translate_batch, caps_txt, src, tgt)
                tr_txt = future.result(timeout=10)  # Timeout after 10 seconds
                caps = [(c[0], c[1], t) for c, t in zip(caps, tr_txt)]

            # Append to SRT file
            last_idx = append_captions_srt(srt_path, start_t, caps, last_idx)
            
            # Store latest captions for real-time display
            if caps:
                session.latest_captions = [{"start": c[0], "end": c[1], "text": c[2]} for c in caps[-3:]]  # Keep last 3 captions
                session.total_captions += len(caps)
                
                # Send update via WebSocket
                asyncio.run(broadcast_to_websockets(session_id, {
                    "type": "new_captions",
                    "count": len(caps),
                    "end_time": end_t,
                    "total_captions": session.total_captions,
                    "captions": session.latest_captions
                }))
            
            log(f"Appended {len(caps)} captions up to {end_t:.1f}s (detected={detected or 'n/a'}, translated_to={tgt or 'off'})", session_id)
            
            # Clean up
            try:
                if os.path.exists(tmp_audio):
                    os.remove(tmp_audio)
            except:
                pass
            
            # Update progress
            if dur > 0:
                session.progress = min(1.0, end_t / dur)
            
            # Check if we should stop
            time.sleep(request.poll_interval)
        
        # Clean up streamlink process if it exists
        if sl_process and sl_process.poll() is None:
            try:
                sl_process.terminate()
            except Exception:
                pass
                
        # Clean up player process if it exists
        if player_proc and player_proc.poll() is None:
            try:
                player_proc.terminate()
            except Exception:
                pass
                
        # Update session status
        session.status = TranscriptionStatus.COMPLETED
        session.end_time = datetime.datetime.now()
        asyncio.run(broadcast_to_websockets(session_id, {
            "type": "status_update",
            "status": session.status,
            "message": "Transcription completed",
            "end_time": session.end_time.isoformat()
        }))
        
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        log(error_msg, session_id)
        session.status = TranscriptionStatus.ERROR
        session.message = error_msg
        asyncio.run(broadcast_to_websockets(session_id, {
            "type": "error",
            "message": error_msg
        }))
    finally:
        # Clean up temporary files
        try:
            # Remove temporary audio files
            for file in os.listdir(session_dir):
                if file.startswith("chunk_") and file.endswith(".wav"):
                    os.remove(os.path.join(session_dir, file))
        except:
            pass

# ---------------------- API Endpoints ----------------------

@app.get("/", response_class=HTMLResponse)
async def get_web_interface(request: Request):
    """Serve a simple web interface"""
    local_ip = get_local_ip()
    return templates.TemplateResponse("index.html", {"request": request, "local_ip": local_ip})

@app.get("/video/{session_id}")
async def video_player(request: Request, session_id: str):
    """Serve the video player page with real-time captions"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    session = active_sessions[session_id]
    youtube_id = extract_youtube_id(session.request.url) if session.request.url else None
    local_ip = get_local_ip()
    
    return templates.TemplateResponse("video_player.html", {
        "request": request,
        "session_id": session_id,
        "youtube_id": youtube_id,
        "session": session,
        "local_ip": local_ip
    })

@app.post("/start")
async def start_transcription(request: TranscriptionRequest):
    """Start a new transcription session"""
    session_id = str(uuid.uuid4())
    
    # Validate request
    if not request.url and not request.file_path:
        return JSONResponse(
            status_code=400,
            content={"error": "Either URL or file_path must be provided"}
        )
    
    # Create session
    session = TranscriptionSession(
        id=session_id,
        status=TranscriptionStatus.STARTING,
        request=request
    )
    active_sessions[session_id] = session
    websocket_connections[session_id] = []
    session_locks[session_id] = threading.Lock()
    
    # Start transcription in background thread
    thread = threading.Thread(
        target=run_transcription_session,
        args=(session_id, request),
        daemon=True
    )
    thread.start()
    session_threads[session_id] = thread
    
    return {"session_id": session_id, "message": "Transcription started"}

@app.post("/stop/{session_id}")
async def stop_transcription(session_id: str):
    """Stop a transcription session"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    active_sessions[session_id].status = TranscriptionStatus.STOPPING
    
    # Clean up WebSocket connections
    if session_id in websocket_connections:
        for ws in websocket_connections[session_id]:
            try:
                await ws.close()
            except:
                pass
        del websocket_connections[session_id]
    
    # Clean up session files in background
    thread_pool.submit(cleanup_session, session_id)
    
    return {"message": "Transcription stopping"}

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get the status of a transcription session"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session.status,
        "progress": session.progress,
        "message": session.message,
        "detected_language": session.detected_language,
        "target_language": session.target_language,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "total_captions": session.total_captions
    }

@app.get("/captions/{session_id}")
async def get_latest_captions(session_id: str):
    """Get the latest captions for a session"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    session = active_sessions[session_id]
    return {
        "captions": session.latest_captions,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/srt/{session_id}")
async def get_srt_file(session_id: str):
    """Get the SRT file for a session"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    srt_path = active_sessions[session_id].srt_path
    if not os.path.exists(srt_path):
        return JSONResponse(
            status_code=404,
            content={"error": "SRT file not found"}
        )
    
    return FileResponse(
        srt_path,
        media_type="text/plain",
        filename=f"captions_{session_id}.srt"
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    # Add to connections
    if session_id not in websocket_connections:
        websocket_connections[session_id] = []
    websocket_connections[session_id].append(websocket)
    
    # Send current status
    session = active_sessions[session_id]
    await websocket.send_json({
        "type": "status_update",
        "status": session.status,
        "message": session.message,
        "progress": session.progress,
        "detected_language": session.detected_language,
        "target_language": session.target_language,
        "total_captions": session.total_captions
    })
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(10)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        # Remove from connections
        if session_id in websocket_connections and websocket in websocket_connections[session_id]:
            websocket_connections[session_id].remove(websocket)
    except Exception as e:
        log(f"WebSocket error: {e}", session_id)
        if session_id in websocket_connections and websocket in websocket_connections[session_id]:
            websocket_connections[session_id].remove(websocket)

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "id": session_id,
                "status": session.status,
                "progress": session.progress,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "total_captions": session.total_captions
            }
            for session_id, session in active_sessions.items()
        ]
    }

@app.get("/network")
async def get_network_info():
    """Get network information for accessing the API"""
    local_ip = get_local_ip()
    return {
        "local_ip": local_ip,
        "access_urls": [
            f"http://{local_ip}:8000",
            f"http://localhost:8000",
            f"http://127.0.0.1:8000"
        ]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its files"""
    if session_id not in active_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    # Stop the session if it's running
    if active_sessions[session_id].status == TranscriptionStatus.RUNNING:
        active_sessions[session_id].status = TranscriptionStatus.STOPPING
    
    # Clean up WebSocket connections
    if session_id in websocket_connections:
        for ws in websocket_connections[session_id]:
            try:
                await ws.close()
            except:
                pass
        del websocket_connections[session_id]
    
    # Remove from active sessions
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    # Clean up session files
    thread_pool.submit(cleanup_session, session_id)
    
    return {"message": "Session deleted"}

# ---------------------- Template Files ----------------------

# Create template files if they don't exist
def create_template_files():
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Transcription API Server</title>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3a0ca3;
                --success: #4cc9f0;
                --danger: #f72585;
                --warning: #fca311;
                --light: #f8f9fa;
                --dark: #212529;
                --gray: #6c757d;
            }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: var(--dark);
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                padding: 25px;
                margin-bottom: 25px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: white;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .form-group { 
                margin-bottom: 20px; 
            }
            
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600;
                color: var(--dark);
            }
            
            input, select { 
                width: 100%; 
                padding: 12px 15px; 
                box-sizing: border-box; 
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            
            input:focus, select:focus {
                border-color: var(--primary);
                outline: none;
                box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
            }
            
            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .checkbox-group input {
                width: auto;
            }
            
            .btn {
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            .btn-primary {
                background: var(--primary);
                color: white;
            }
            
            .btn-primary:hover {
                background: var(--secondary);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .btn-danger {
                background: var(--danger);
                color: white;
            }
            
            .btn-danger:hover {
                background: #e50c75;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .btn-success {
                background: var(--success);
                color: white;
            }
            
            .btn-success:hover {
                background: #3ab7d8;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .status {
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-weight: 500;
            }
            
            .running { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
            .error { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
            .idle { background: #e2e3e5; color: #383d41; border-left: 4px solid #6c757d; }
            .starting { background: #cce5ff; color: #004085; border-left: 4px solid #007bff; }
            
            .logs {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                height: 250px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                border: 1px solid #e9ecef;
            }
            
            .log-entry {
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }
            
            .log-time {
                color: var(--gray);
                margin-right: 10px;
            }
            
            .network-info {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 25px;
            }
            
            .network-info h3 {
                margin-top: 0;
                font-size: 1.5rem;
            }
            
            .url-list {
                list-style: none;
                padding: 0;
            }
            
            .url-list li {
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .url-list a {
                color: white;
                text-decoration: none;
                font-weight: 500;
            }
            
            .url-list a:hover {
                text-decoration: underline;
            }
            
            .video-link {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 15px;
                background: var(--success);
                color: white;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                margin-top: 20px;
                transition: all 0.3s;
            }
            
            .video-link:hover {
                background: #3ab7d8;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .stat-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .stat-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--primary);
                margin: 10px 0;
            }
            
            .stat-label {
                font-size: 0.9rem;
                color: var(--gray);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .stats {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Live Transcription API Server</h1>
                <p>Real-time transcription and translation for live streams</p>
            </div>
            
            <div class="network-info">
                <h3>Network Access Information</h3>
                <p>This server is accessible at:</p>
                <ul class="url-list">
                    <li><a href="http://{{ local_ip }}:8000" target="_blank">http://{{ local_ip }}:8000</a> (Local Network)</li>
                    <li><a href="http://localhost:8000" target="_blank">http://localhost:8000</a> (This Device)</li>
                </ul>
                <p>Other devices on your network can access this interface using the first URL.</p>
            </div>
            
            <div class="card">
                <h2>Transcription Settings</h2>
                
                <div class="form-group">
                    <label for="url">Media URL (YouTube, Twitch, etc.):</label>
                    <input type="text" id="url" placeholder="https://youtube.com/watch?v=...">
                </div>
                
                <div class="form-group">
                    <label for="model">Model Size:</label>
                    <select id="model">
                        <option value="tiny">Tiny (fastest, least accurate)</option>
                        <option value="base">Base</option>
                        <option value="small" selected>Small (recommended)</option>
                        <option value="medium">Medium</option>
                        <option value="large">Large (slowest, most accurate)</option>
                    </select>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="auto_translate" checked>
                    <label for="auto_translate">Enable Auto Translation</label>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="low_latency" checked>
                    <label for="low_latency">Low Latency Mode (faster but less accurate)</label>
                </div>
                
                <div class="form-group checkbox-group">
                    <input type="checkbox" id="play">
                    <label for="play">Launch Media Player (Windows only)</label>
                </div>
                
                <button class="btn btn-primary" onclick="startTranscription()">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M11.596 8.697l-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/>
                    </svg>
                    Start Transcription
                </button>
                <button class="btn btn-danger" onclick="stopTranscription()">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M5 3.5h6A1.5 1.5 0 0 1 12.5 5v6a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 11V5A1.5 1.5 0 0 1 5 3.5z"/>
                    </svg>
                    Stop Transcription
                </button>
                
                <div id="status" class="status idle">Status: Idle</div>
                
                <div class="stats" id="stats" style="display: none;">
                    <div class="stat-card">
                        <div class="stat-label">Total Captions</div>
                        <div class="stat-value" id="total-captions">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Detected Language</div>
                        <div class="stat-value" id="detected-language">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Target Language</div>
                        <div class="stat-value" id="target-language">-</div>
                    </div>
                </div>
                
                <a class="video-link" id="videoLink" style="display: none;" target="_blank">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2V5z"/>
                    </svg>
                    Open Video Player with Live Captions
                </a>
            </div>
            
            <div class="card">
                <h2>Activity Log</h2>
                <div class="logs" id="logs"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let sessionId = null;
            
            function addLog(message) {
                const logs = document.getElementById('logs');
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.innerHTML = `
                    <span class="log-time">${new Date().toLocaleTimeString()}</span>
                    <span class="log-message">${message}</span>
                `;
                logs.appendChild(logEntry);
                logs.scrollTop = logs.scrollHeight;
            }
            
            function updateStatus(status, message) {
                const statusDiv = document.getElementById('status');
                statusDiv.className = `status ${status.toLowerCase()}`;
                statusDiv.innerHTML = `<strong>Status:</strong> ${status} - ${message}`;
            }
            
            function updateStats(data) {
                document.getElementById('stats').style.display = 'grid';
                if (data.total_captions !== undefined) {
                    document.getElementById('total-captions').textContent = data.total_captions;
                }
                if (data.detected_language) {
                    document.getElementById('detected-language').textContent = data.detected_language;
                }
                if (data.target_language) {
                    document.getElementById('target-language').textContent = data.target_language;
                }
            }
            
            function connectWebSocket() {
                if (ws) {
                    ws.close();
                }
                
                ws = new WebSocket('ws://' + window.location.host + '/ws/' + sessionId);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'status_update') {
                        updateStatus(data.status, data.message);
                        updateStats(data);
                        
                        // Show video link when transcription starts
                        if (data.status === 'running') {
                            document.getElementById('videoLink').style.display = 'flex';
                            document.getElementById('videoLink').href = '/video/' + sessionId;
                        }
                    } else if (data.type === 'new_captions') {
                        addLog('Added ' + data.count + ' new captions (Total: ' + data.total_captions + ')');
                        updateStats({total_captions: data.total_captions});
                    } else if (data.type === 'language_detected') {
                        addLog('Detected language: ' + data.detected_name + ', translating to: ' + data.target_name);
                        updateStats({
                            detected_language: data.detected_name,
                            target_language: data.target_name
                        });
                    } else if (data.type === 'error') {
                        updateStatus('error', data.message);
                    }
                };
                
                ws.onopen = function() {
                    addLog('WebSocket connection established');
                };
                
                ws.onclose = function() {
                    addLog('WebSocket connection closed');
                };
            }
            
            function startTranscription() {
                const url = document.getElementById('url').value;
                const model = document.getElementById('model').value;
                const autoTranslate = document.getElementById('auto_translate').checked;
                const lowLatency = document.getElementById('low_latency').checked;
                const play = document.getElementById('play').checked;
                
                if (!url) {
                    alert('Please enter a URL');
                    return;
                }
                
                fetch('/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        url: url,
                        model: model,
                        auto_translate: autoTranslate,
                        low_latency: lowLatency,
                        play: play
                    })
                })
                .then(response => response.json())
                .then(data => {
                    sessionId = data.session_id;
                    addLog('Session started: ' + sessionId);
                    updateStatus('starting', 'Session is starting...');
                    connectWebSocket();
                })
                .catch(error => {
                    addLog('Error: ' + error);
                });
            }
            
            function stopTranscription() {
                if (!sessionId) {
                    addLog('No active session');
                    return;
                }
                
                fetch('/stop/' + sessionId, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    addLog('Session stopped: ' + data.message);
                    updateStatus('idle', 'No active session');
                    document.getElementById('videoLink').style.display = 'none';
                    document.getElementById('stats').style.display = 'none';
                })
                .catch(error => {
                    addLog('Error: ' + error);
                });
            }
        </script>
    </body>
    </html>
    """
    
    # Create video_player.html
    video_player_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Video with Real-time Translation</title>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3a0ca3;
                --success: #4cc9f0;
                --danger: #f72585;
                --dark: #121212;
                --light: #f8f9fa;
            }
            
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 0; 
                background: var(--dark);
                color: var(--light);
                overflow-x: hidden;
            }
            
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 20px;
                padding: 20px;
                background: rgba(0, 0, 0, 0.5);
                border-radius: 12px;
            }
            
            .header h1 {
                margin: 0;
                font-size: 2.2rem;
                background: linear-gradient(45deg, var(--primary), var(--success));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .video-container { 
                position: relative; 
                width: 100%; 
                margin-bottom: 20px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            #player { 
                width: 100%; 
                height: 500px; 
                background: #000;
            }
            
            .captions-container { 
                position: absolute; 
                bottom: 20px; 
                left: 0; 
                width: 100%; 
                text-align: center; 
                padding: 15px;
                box-sizing: border-box;
                z-index: 10;
            }
            
            #captions { 
                font-size: 24px; 
                font-weight: 600;
                color: #fff; 
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
                margin: 0;
                padding: 15px 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 8px;
                display: inline-block;
                max-width: 90%;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .controls { 
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            .btn-primary {
                background: var(--primary);
                color: white;
            }
            
            .btn-primary:hover {
                background: var(--secondary);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            .btn-success {
                background: var(--success);
                color: white;
            }
            
            .btn-success:hover {
                background: #3ab7d8;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            .status {
                text-align: center;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                font-weight: 500;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(5px);
            }
            
            .running { 
                color: #4ade80; 
                border: 1px solid rgba(74, 222, 128, 0.3);
            }
            
            .error { 
                color: #f87171; 
                border: 1px solid rgba(248, 113, 113, 0.3);
            }
            
            .idle { 
                color: #9ca3af; 
                border: 1px solid rgba(156, 163, 175, 0.3);
            }
            
            .stats { 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .stat-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--primary);
                margin: 10px 0;
            }
            
            .stat-label {
                font-size: 0.9rem;
                color: #9ca3af;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                #player {
                    height: 300px;
                }
                
                #captions {
                    font-size: 18px;
                    padding: 10px 15px;
                }
                
                .stats {
                    grid-template-columns: 1fr;
                }
                
                .header h1 {
                    font-size: 1.8rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Live Video with Real-time Translation</h1>
            </div>
            
            <div class="video-container">
                {% if youtube_id %}
                <div id="player"></div>
                {% else %}
                <div style="text-align: center; padding: 50px; background: #333; border-radius: 12px;">
                    <p>Video playback not available for this URL</p>
                    <p>Transcription is still running in the background</p>
                </div>
                {% endif %}
                <div class="captions-container">
                    <p id="captions">Waiting for captions...</p>
                </div>
            </div>
            
            <div class="status" id="status">Status: {{ session.status.value }}</div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Total Captions</div>
                    <div class="stat-value" id="total-captions">{{ session.total_captions }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Detected Language</div>
                    <div class="stat-value" id="detected-language">{{ session.detected_language or '-' }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Target Language</div>
                    <div class="stat-value" id="target-language">{{ session.target_language or '-' }}</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="goBack()">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path fill-rule="evenodd" d="M15 8a.5.5 0 0 0-.5-.5H2.707l3.147-3.146a.5.5 0 1 0-.708-.708l-4 4a.5.5 0 0 0 0 .708l4 4a.5.5 0 0 0 .708-.708L2.707 8.5H14.5A.5.5 0 0 0 15 8z"/>
                    </svg>
                    Back to Control Panel
                </button>
                <button class="btn btn-success" onclick="downloadSRT()">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                        <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
                    </svg>
                    Download SRT File
                </button>
            </div>
        </div>
        
        {% if youtube_id %}
        <script>
            // Load YouTube IFrame API
            var tag = document.createElement('script');
            tag.src = "https://www.youtube.com/iframe_api";
            var firstScriptTag = document.getElementsByTagName('script')[0];
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
            
            var player;
            function onYouTubeIframeAPIReady() {
                player = new YT.Player('player', {
                    height: '500',
                    width: '100%',
                    videoId: '{{ youtube_id }}',
                    playerVars: {
                        'playsinline': 1,
                        'autoplay': 1,
                        'controls': 1,
                        'modestbranding': 1,
                        'rel': 0
                    },
                    events: {
                        'onReady': onPlayerReady,
                        'onStateChange': onPlayerStateChange
                    }
                });
            }
            
            function onPlayerReady(event) {
                event.target.playVideo();
            }
            
            function onPlayerStateChange(event) {
                // Handle player state changes if needed
            }
        </script>
        {% endif %}
        
        <script>
            const sessionId = '{{ session_id }}';
            let ws = null;
            let captionsElement = document.getElementById('captions');
            let statusElement = document.getElementById('status');
            let totalCaptionsElement = document.getElementById('total-captions');
            let detectedLanguageElement = document.getElementById('detected-language');
            let targetLanguageElement = document.getElementById('target-language');
            
            // Connect to WebSocket for real-time updates
            function connectWebSocket() {
                ws = new WebSocket('ws://' + window.location.host + '/ws/' + sessionId);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'status_update') {
                        statusElement.textContent = 'Status: ' + data.status;
                        statusElement.className = 'status ' + data.status;
                        
                        if (data.total_captions !== undefined) {
                            totalCaptionsElement.textContent = data.total_captions;
                        }
                        if (data.detected_language) {
                            detectedLanguageElement.textContent = data.detected_language;
                        }
                        if (data.target_language) {
                            targetLanguageElement.textContent = data.target_language;
                        }
                    } else if (data.type === 'new_captions' && data.captions && data.captions.length > 0) {
                        // Display the latest caption
                        const latestCaption = data.captions[data.captions.length - 1];
                        captionsElement.textContent = latestCaption.text;
                        
                        // Update stats
                        if (data.total_captions !== undefined) {
                            totalCaptionsElement.textContent = data.total_captions;
                        }
                    } else if (data.type === 'language_detected') {
                        detectedLanguageElement.textContent = data.detected_name;
                        targetLanguageElement.textContent = data.target_name;
                    }
                };
                
                ws.onclose = function() {
                    // Try to reconnect after 2 seconds
                    setTimeout(connectWebSocket, 2000);
                };
            }
            
            // Poll for captions every 100ms as a fallback
            function startCaptionPolling() {
                setInterval(async () => {
                    try {
                        const response = await fetch('/captions/' + sessionId);
                        const data = await response.json();
                        
                        if (data.captions && data.captions.length > 0) {
                            const latestCaption = data.captions[data.captions.length - 1];
                            captionsElement.textContent = latestCaption.text;
                        }
                    } catch (error) {
                        console.error('Error fetching captions:', error);
                    }
                }, 100); // Poll every 100 milliseconds
            }
            
            function goBack() {
                window.location.href = '/';
            }
            
            function downloadSRT() {
                window.open('/srt/' + sessionId, '_blank');
            }
            
            // Initialize
            connectWebSocket();
            startCaptionPolling();
        </script>
    </body>
    </html>
    """
    
    # Write template files
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open("templates/video_player.html", "w", encoding="utf-8") as f:
        f.write(video_player_html)

# ---------------------- Main Entry Point ----------------------

if __name__ == "__main__":
    import uvicorn
    
    # Create template files
    create_template_files()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live Transcribe & Auto-Translate API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Display network information
    local_ip = get_local_ip()
    print("=" * 60)
    print("Live Transcription API Server Starting...")
    print(f"Local access: http://localhost:{args.port}")
    print(f"Network access: http://{local_ip}:{args.port}")
    print("Other devices can access the web interface using the network URL")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
