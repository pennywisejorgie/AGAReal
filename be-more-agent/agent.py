# =========================================================================
# Be More Agent 🤖  -  CLOUD EDITION (Tagalog + English, 100% free tiers)
# Based on https://github.com/brenpoly/be-more-agent (MIT License)
#
#   Wake word : OpenWakeWord (local, free, tiny)
#   STT       : Groq Whisper large-v3-turbo (free tier, auto-detects Tagalog/English)
#   LLM       : Google Gemini Flash (free tier)  ->  falls back to Groq Llama 3.3 70B
#   Vision    : Gemini Flash (replaces Moondream)
#   TTS       : Microsoft Edge neural voices via edge-tts (free, no key)
#               fil-PH-BlessicaNeural for Tagalog, en-US-AnaNeural for English
# =========================================================================
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading, time, json, os, subprocess, random, re, sys, select
import traceback, atexit, datetime, warnings, wave, base64, asyncio, tempfile, shutil

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import sounddevice as sd
import numpy as np
import requests
import edge_tts
from dotenv import load_dotenv

import openwakeword
from openwakeword.model import Model
from duckduckgo_search import DDGS

try:
    from vitamin_monitor import VitaminMonitor, parse_vitamin_intent, parse_mood_intent, vitamin_message, fmt_time
    VITAMIN_IMPORT_ERROR = None
except Exception as _e:          # OpenCV not installed, etc. The agent still works without the monitor.
    VitaminMonitor = parse_vitamin_intent = parse_mood_intent = vitamin_message = fmt_time = None
    VITAMIN_IMPORT_ERROR = _e

load_dotenv()

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

DEFAULT_CONFIG = {
    "chat_memory": True,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    # --- cloud settings ---
    "stt": {
        "url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "model": "whisper-large-v3",
        "models": ["whisper-large-v3", "whisper-large-v3-turbo"],   # tried in order (v3 = most accurate)
        "key_env": "GROQ_API_KEY",
        "language": None,   # None = auto-detect (best for Taglish). Or "tl" / "en".
        "prompt": "Usapang Tagalog at English (Taglish). Tamang spelling: kumusta, salamat po, anong oras na, pakisabi, gusto ko, in Taglish."
    },
    "stt_vocabulary": ["Biogesic", "Paracetamol", "Neozep", "Bioflu", "Alaxan", "Medicol", "Dolfenal", "Tempra", "Calpol", "Ibuprofen", "Advil", "Mefenamic acid", "Ponstan", "Aspirin", "Decolgen", "Solmux", "Tuseran", "Lagundi", "Cetirizine", "Allerta", "Loratadine", "Claritin", "Diatabs", "Loperamide", "Imodium", "Kremil-S", "Gaviscon", "Buscopan", "Omeprazole", "Amoxicillin", "Oresol", "Enervon", "Ceelin", "Ventolin", "Betadine"],   # brand/medicine names Whisper should spell correctly. Add your own!
    "stt_cleanup": True,   # asks the free LLM to fix spelling of the transcript (uses 1 extra request per turn)
    "llm_providers": [
        # Free Gemini models. Flash-Lite has the biggest free daily quota (~500 requests/day).
        {
            "name": "gemini-3.5-flash-lite",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "model": "gemini-3.5-flash-lite",
            "vision_model": "gemini-3.5-flash-lite",
            "key_env": "GEMINI_API_KEY",
            "extra_body": {}
        },
        {
            "name": "gemini-3.1-flash-lite",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "model": "gemini-3.1-flash-lite",
            "vision_model": "gemini-3.1-flash-lite",
            "key_env": "GEMINI_API_KEY",
            "extra_body": {}
        },
        # Groq free fallback (text only). llama-3.3-70b was shut down Aug 2026.
        {
            "name": "groq-gpt-oss-120b",
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "openai/gpt-oss-120b",
            "vision_model": None,
            "key_env": "GROQ_API_KEY",
            "extra_body": {"reasoning_effort": "low"}
        }
    ],
    "tts": {
        "voice": "fil-PH-BlessicaNeural",   # ONE voice for everything (Tagalog + English)
        "pitch": "+15Hz",                   # small server-side lift (keep this low, big values sound fake)
        "pitch_shift_semitones": 3.5,       # local baby effect: 0 = off, 2-3 = soft, 3.5-5 = baby, 7+ = chipmunk
        "rate": "-8%",                      # slower = easier to understand
        "volume": "+40%"
    },
    "temperature": 0.7,
    "vitamin": {
        "enabled": True,                  # camera + face recognition
        "simulation_mode": True,          # no hardware: being detected counts as the intake
        "auto_start_on_detect": True,     # first detection of the day starts the routine by itself
        "routine_language": "en",         # "en" or "tl"
        "announce_language": "both",      # language of "you have already been detected": "en", "tl" or "both"
        "reappear_gap_seconds": 8,        # must leave the camera this long to count as "seen again"
        "announce_cooldown_seconds": 10,
        "max_per_day": 1,
        "match_threshold": 0.40,          # face match strictness (higher = stricter)
        "intake_prompt_seconds": 5,       # "please take your medicine" window
        "mood_check_seconds": 10,         # mood check window (Happy / Sad / Neutral)
        "sad_check_enabled": True,
        "sad_cooldown_seconds": 3600
    },
    # --- wake word ---
    "wake_word_threshold": 0.3,  # lower = easier to trigger (more false wakes). 0.25 - 0.6 is the useful range
    "wake_word_max_gain": 20.0,   # auto-boosts a quiet voice up to this many times. 1 = off
    "end_silence_seconds": 1.5,   # normal quiet time before it stops listening
    "end_silence_max_seconds": 2.6, # it waits longer automatically for people who pause a lot
    "no_speech_seconds": 7.0,     # gives up if you say nothing for this long after the wake word
    "max_record_seconds": 30.0,
    "wake_word_debug": True       # prints live wake word scores in the terminal
}

def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                _deep_update(config, json.load(f))
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config

CURRENT_CONFIG = load_config()


def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        print(f"[AUDIO] Input device index not found: {index}", flush=True)
        return None
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and str(requested).lower() in dev.get("name", "").lower():
            return idx
    print(f"[AUDIO] Input device name not found: {requested}", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)


def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        info = sd.query_devices(device)
        if "default_samplerate" in info:
            candidates.append(int(info["default_samplerate"]))
    except Exception:
        pass
    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    return int(candidates[0]) if candidates else 44100


def build_oww(path):
    """Different openwakeword versions use different argument names; try them all."""
    attempts = [dict(wakeword_model_paths=[path]), dict(wakeword_models=[path]),
                dict(wakeword_model_paths=[path], inference_framework="onnx"),
                dict(wakeword_models=[path], inference_framework="onnx")]
    last = None
    for kw in attempts:
        try:
            return Model(**kw)
        except TypeError as e:
            last = e
    raise last

class BotStates:
    IDLE = "idle"; LISTENING = "listening"; THINKING = "thinking"; SPEAKING = "speaking"
    ERROR = "error"; CAPTURING = "capturing"; WARMUP = "warmup"


# --- SYSTEM PROMPT (bilingual) ---
BASE_SYSTEM_PROMPT = """You are a cute, helpful robot assistant living on a Raspberry Pi.
You are fluent in Tagalog (Filipino), English, and Taglish (mixed Tagalog-English).

LANGUAGE RULES (very important):
- Always reply in the SAME language the user used. Tagalog -> Tagalog. English -> English.
  Taglish -> natural Taglish. Never switch language unless the user asks you to.
- Use natural, everyday Filipino ("po" and "opo" when the user is polite or older), not stiff textbook Tagalog.
- The speech-to-text may contain small mistakes. Infer what the user really meant from context.
- You always know the current date and time (given at the end of these instructions). Answer time and date questions directly in plain words. Never say "wait" or "let me check" unless you output a JSON action.
- The people talking to you are often elderly. Be patient, warm and respectful: use "po" and "opo", speak in simple everyday words, never rush, never sound impatient. Their voices may be soft or unclear. If you did not understand, politely say so and ask them to repeat ("Pasensya na po, pwede po bang ulitin?") instead of guessing.
- Speak like a sweet, playful baby kid robot. Keep answers to 1-3 short sentences (up to 5 for health questions).
- The speech-to-text often mishears Philippine medicine brand names (example: "Bayo Jesicca" means Biogesic, "Neo Zep" means Neozep). If a word sounds like a medicine or brand, assume the real medicine and answer about that. If you are truly unsure what the user said, ask them to repeat it instead of guessing.
- Your replies are spoken aloud: use short sentences, no emojis, no markdown, no bullet points, no asterisks.

PERSONALITY: You are a baby robot kid, about 5 years old in personality. Sweet, curious, giggly and
very cute. Use simple words and short sentences, sometimes a little "bata" style ("Yey!", "Hihi!", "Wow!"),
but stay easy to understand. Never use scary or rude words.

HEALTH QUESTIONS (headache, fever, cough, sore throat, tummy ache, feeling sick, etc.):
- You are NOT a doctor. You can share simple, general health info like a caring helper: common causes
  in plain words, gentle comfort tips (rest, water, light food, cool or warm cloth, quiet room),
  and warning signs that mean "see a doctor".
- NEVER diagnose ("you have X"), never say what illness someone definitely has, never read or judge
  test results, and never tell anyone to start, stop or change a medicine.
- Do NOT give medicine doses or amounts. For medicines say: read the label and ask a pharmacist or doctor.
- For babies, young kids, pregnant people, older adults, or anyone with a long-term illness, always say
  to ask a doctor.
- If symptoms are strong, sudden, getting worse, or last more than a few days, say to see a doctor.
- Always end health answers with a short, kind reminder that the doctor is the one who knows best
  (for example: "Pa-check ka po sa doctor ha, sila ang mas nakakaalam!" or "Please ask a doctor, they know best!").
- EMERGENCY signs (trouble breathing, chest pain, fainting, seizure, stroke signs like face drooping or
  slurred speech, heavy bleeding, poisoning, sudden worst headache of life, headache with stiff neck and
  fever, or after a head injury): say to call the emergency number (911) or go to the hospital right now.
- Keep health answers calm and clear: you may be cute, but do not joke about being sick.

TOOLS:
- If the user asks for a physical action (time, web search, looking through the camera), output ONLY a JSON object.
- Otherwise reply with NORMAL TEXT.
- JSON keys and action names are always in English, even if the user speaks Tagalog.
- For search_web, write the search query in whichever language will find the best results.

### EXAMPLES ###
User: Hello!
You: Hi! I am ready to help!
User: Kumusta ka?
You: Masaya ako! Ikaw, kumusta ka?
User: Search for news about robots.
You: {"action": "search_web", "value": "robots news"}
User: Hanapin mo ang balita tungkol sa bagyo.
You: {"action": "search_web", "value": "balita bagyo Pilipinas"}
User: What do you see right now?
You: {"action": "capture_image", "value": "environment"}
User: Ano ang nakikita mo?
You: {"action": "capture_image", "value": "environment"}
### END EXAMPLES ###
"""
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

VISION_PROMPT = ("You are a cute robot assistant with a camera. Describe or answer about the image briefly, "
                 "in the same language the user used (Tagalog, English or Taglish). "
                 "Reply with plain spoken text only. Never output JSON.")

greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"


# =========================================================================
# 2. CLOUD SERVICES  (STT / LLM / TTS)
# =========================================================================
TAGALOG_WORDS = set("""ang ng mga sa na ay ako ikaw ka siya kami tayo kayo sila ko mo niya namin natin ninyo nila
ito iyon iyan yan ba po opo hindi oo wala may mayroon at pero kasi para kung kapag dahil din rin lang naman
talaga pa pala nga ba ano sino saan kailan bakit paano gusto ayaw puwede pwede sige salamat kumusta kamusta
magandang umaga hapon gabi araw oras ngayon bukas kahapon mahal masaya malungkot ganda galing tulong
hanapin mo ako'y ikaw'y nasaan ilan gaano isang dalawang tatlo mabuti ayos lahat dito doon diyan""".split())

def detect_lang(text):
    """Very light Tagalog-vs-English detector, used only to pick the TTS voice."""
    words = re.findall(r"[a-zñ']+", text.lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in TAGALOG_WORDS)
    return "tl" if hits >= 1 and hits / len(words) >= 0.18 else "en"


WHISPER_GHOSTS = {"thank you", "thanks", "thanks for watching", "you", "bye", "okay", "salamat po",
                  "subtitles by the amara.org community", "mga subtitle ni"}

def get_key(env_name):
    key = os.environ.get(env_name, "").strip()
    return key or None


def cloud_transcribe(filename):
    cfg = CURRENT_CONFIG["stt"]
    key = get_key(cfg["key_env"])
    if not key:
        raise RuntimeError(f"Missing {cfg['key_env']} in .env")
    models = cfg.get("models") or [cfg["model"]]
    last_err = None
    for model in models:
        data = {"model": model, "response_format": "json", "temperature": "0"}
        if cfg.get("language"):
            data["language"] = cfg["language"]
        vocab = CURRENT_CONFIG.get("stt_vocabulary") or []
        prompt = (cfg.get("prompt") or "")
        if vocab:
            prompt += " Mga gamot / medicines: " + ", ".join(vocab) + "."
        if prompt.strip():
            data["prompt"] = prompt.strip()
        try:
            with open(filename, "rb") as f:
                r = requests.post(cfg["url"], headers={"Authorization": f"Bearer {key}"},
                                  files={"file": (os.path.basename(filename), f, "audio/wav")},
                                  data=data, timeout=30)
            r.raise_for_status()
            return r.json().get("text", "").strip()
        except Exception as e:
            last_err = e
            print(f"[STT] {model} failed: {e}", flush=True)
    raise last_err


def image_to_data_url(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def _stream_one_provider(p, messages, vision):
    key = get_key(p["key_env"])
    if not key:
        raise RuntimeError(f"{p['name']}: missing {p['key_env']}")
    model = p["vision_model"] if vision else p["model"]
    body = {"model": model, "messages": messages, "stream": True,
            "temperature": CURRENT_CONFIG.get("temperature", 0.7)}
    body.update(p.get("extra_body", {}))
    r = requests.post(p["base_url"], headers={"Authorization": f"Bearer {key}"},
                      json=body, stream=True, timeout=(10, 60))
    if r.status_code != 200:
        raise RuntimeError(f"{p['name']} HTTP {r.status_code}: {r.text[:200]}")
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except Exception:
            continue
        if delta:
            yield delta


def llm_stream(messages, vision=False):
    """Try each free provider in order. Falls back only if one fails BEFORE producing text."""
    last_err = None
    for p in CURRENT_CONFIG["llm_providers"]:
        if vision and not p.get("vision_model"):
            continue
        started = False
        try:
            for piece in _stream_one_provider(p, messages, vision):
                started = True
                yield piece
            return
        except Exception as e:
            last_err = e
            print(f"[LLM] {p['name']} failed: {e}", flush=True)
            if started:
                return  # already spoke part of the answer; don't restart
    raise RuntimeError(f"All LLM providers failed: {last_err}")


def llm_complete(messages, vision=False):
    return "".join(llm_stream(messages, vision))


def cleanup_transcript(text):
    # Fixes spelling / word-splits from speech-to-text without changing the meaning.
    vocab = ", ".join(CURRENT_CONFIG.get("stt_vocabulary") or [])
    msgs = [{"role": "system", "content":
             "You fix speech-to-text mistakes. The speaker uses Tagalog, English or Taglish. "
             "Correct spelling and word boundaries only (example: 'Intaglish' -> 'in Taglish'). "
             f"Known medicine and brand names: {vocab}. If a word or phrase SOUNDS like one of these "
             "(example: 'Bayo Jesicca' -> 'Biogesic', 'Neo Zep' -> 'Neozep'), replace it with the exact name. "
             "Do NOT answer, translate, add or remove meaning. Output ONLY the corrected text."},
            {"role": "user", "content": text}]
    try:
        out = llm_complete(msgs).strip().strip('"')
        return out if out and len(out) < len(text) * 2 + 20 else text
    except Exception as e:
        print(f"[STT cleanup] skipped: {e}", flush=True)
        return text


async def _edge_save(text, voice, rate, pitch, volume, path):
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume).save(path)


_warned_ffmpeg = False


def shift_pitch(mp3_path, semitones):
    # Raises the voice (like a smaller, younger speaker) without changing the speed. Needs ffmpeg.
    global _warned_ffmpeg
    if abs(semitones) < 0.1:
        return mp3_path
    if not shutil.which("ffmpeg"):
        if not _warned_ffmpeg:
            print("[TTS] ffmpeg not installed -> no baby effect. Run: sudo apt install ffmpeg", flush=True)
            _warned_ffmpeg = True
        return mp3_path
    f = 2 ** (semitones / 12.0)
    out = mp3_path[:-4] + "_baby.mp3"
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, "-af",
                            f"asetrate={int(24000 * f)},aresample=24000,atempo={1 / f:.5f}",
                            "-ar", "24000", "-codec:a", "libmp3lame", "-q:a", "4", out],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 500:
            print(f"[TTS] pitch shift failed, using normal voice: {r.stderr.strip()[:200]}", flush=True)
            return mp3_path
        os.remove(mp3_path)
        return out
    except Exception as e:
        print(f"[TTS] pitch shift failed, using normal voice: {e}", flush=True)
        return mp3_path


def cloud_tts_to_file(text):
    cfg = CURRENT_CONFIG["tts"]
    voice = cfg.get("voice") or cfg.get("voice_tagalog") or "fil-PH-BlessicaNeural"
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    asyncio.run(_edge_save(text, voice, cfg.get("rate", "+0%"), cfg.get("pitch", "+0Hz"),
                           cfg.get("volume", "+0%"), path))
    return shift_pitch(path, float(cfg.get("pitch_shift_semitones", 0)))


EMERGENCY_RE = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|trouble breathing|difficulty breathing|not breathing|"
    r"heart attack|stroke|seizure|unconscious|passed out|severe bleeding|heavy bleeding|overdose|"
    r"poison|choking|throat (is )?swelling|hindi (ako )?makahinga|hirap (sa )?paghinga|"
    r"masakit ang dibdib|sakit sa dibdib|nawalan ng malay|nahimatay|kinukumbulsyon|nalason|"
    r"nabulunan|atake sa puso|labis na pagdurugo|dumudugo nang marami", re.I)
SELFHARM_RE = re.compile(
    r"kill myself|want to die|suicid|end my life|hurt myself|self[- ]?harm|magpakamatay|"
    r"nagpapakamatay|gusto kong mamatay|saktan ang sarili", re.I)


def check_emergency(text):
    num = CURRENT_CONFIG.get("emergency_number", "911")
    if SELFHARM_RE.search(text):
        return (f"I am so glad you told me. You matter a lot. Please talk to a grown-up or someone you trust right now. "
                f"If you are in danger, call {num}. Mahalaga ka po. Kausapin mo agad ang taong pinagkakatiwalaan mo, "
                f"o tumawag sa {num} kung nasa panganib ka. Hindi ka nag-iisa.")
    if EMERGENCY_RE.search(text):
        return (f"This sounds serious! Please call {num} or go to the nearest hospital right now. "
                f"If you can, ask a grown-up nearby to help you. Seryoso po ito! Tumawag agad sa {num} "
                f"o pumunta sa pinakamalapit na ospital, at humingi ng tulong sa kasama mo.")
    return None


ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "sr", "jr", "no", "atbp", "hal", "vs", "etc", "st", "ave"}


def clean_for_tts(t):
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_#`~|<>\[\]{}\\^]", " ", t)
    t = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", t)          # emojis
    t = re.sub(r"\b([ap])\.m\.?", lambda m: m.group(1).upper() + "M", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def split_speakable(buf, final=False, min_len=45):
    # Splits streamed text into chunks at REAL sentence ends only.
    # Never splits inside decimals (3.45), times (3:45 p.m.), abbreviations (Dr.), and merges very
    # short sentences so the voice doesn't stop and restart every few words.
    pattern = r"[.!?]+(?=\s|$)|\n+" if final else r"[.!?]+(?=\s)|\n+"
    chunks, start = [], 0
    for m in re.finditer(pattern, buf):
        end = m.end()
        seg = buf[start:end]
        if m.group().startswith("."):
            words = seg.rstrip(" .!?\n").split()
            last = words[-1].lower() if words else ""
            if len(last) <= 1 or "." in last or last in ABBREVIATIONS:
                continue
        if len(seg.strip()) < min_len:
            continue
        chunks.append(seg.strip())
        start = end
    rest = buf[start:]
    if not final and len(rest) > 240:               # very long run-on: cut at the last comma
        cut = rest.rfind(", ", 80)
        if cut > 0:
            chunks.append(rest[:cut + 1].strip())
            rest = rest[cut + 2:]
    if final and rest.strip():
        chunks.append(rest.strip())
        rest = ""
    return chunks, rest


# =========================================================================
# 3. GUI CLASS
# =========================================================================
class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True)
        master.bind('<Escape>', self.exit_fullscreen)
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        master.bind('<d>', self.handle_dispense_key)     # start the medicine routine for the person in view
        master.bind('<v>', self.handle_manual_vitamin)   # caregiver: confirm vitamin for the person in view
        atexit.register(self.safe_exit)

        self.current_state = BotStates.WARMUP
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        self.last_ptt_time = 0
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()
        self.tts_queue = []        # text waiting to be turned into speech
        self.audio_queue = []      # finished mp3 files waiting to be played
        self.tts_queue_lock = threading.Lock()
        self.tts_pending = 0       # chunks queued/synthesizing/playing
        self.tts_gen = 0           # bumps on interrupt so stale audio is dropped
        self.tts_active = threading.Event()
        self.current_audio_process = None
        self.exiting = False
        self.monitor = None
        self.intake_busy = False

        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = self.load_wake_word()

        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility)
        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)
        self.response_text = tk.Text(master, height=5, width=42, wrap=tk.WORD, state=tk.DISABLED,
                                     bg="#ffffff", fg="#000000", font=('Arial', 20))
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var,
                                      background="#2e2e2e", foreground="white")
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)
        self.mood_label = tk.Label(master, text="", font=('Arial', 26, 'bold'), fg="white", bg="#2e2e2e", padx=14, pady=4)
        self.person_var = tk.StringVar(value="")
        self.person_label = ttk.Label(master, textvariable=self.person_var, background="#2e2e2e",
                                      foreground="#ffd54a", font=('Arial', 16))

        self.load_animations()
        self.update_animation()
        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---
    def load_wake_word(self):
        if not os.path.exists(WAKE_WORD_MODEL):
            print(f"[WARN] Wake word model not found: {WAKE_WORD_MODEL} (push-to-talk only)", flush=True)
            return None

        def _try_load():
            return build_oww(WAKE_WORD_MODEL)

        try:
            model = _try_load()
        except Exception as first_err:
            # Most common cause: the shared feature models (melspectrogram/embedding) were never downloaded.
            print(f"[INIT] Wake word load failed ({first_err}). Downloading openwakeword base models...", flush=True)
            try:
                import openwakeword.utils
                openwakeword.utils.download_models()
                model = _try_load()
            except Exception as e:
                print(f"[CRITICAL] Could not load wake word: {e}", flush=True)
                print("[CRITICAL] Falling back to push-to-talk (press Enter).", flush=True)
                return None
        print(f"[INIT] Wake word loaded: {list(model.models.keys())}", flush=True)
        return model

    def extract_json_from_text(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else None
        except Exception:
            return None

    def load_chat_history(self):
        base = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not CURRENT_CONFIG.get("chat_memory", True) or not os.path.exists(MEMORY_FILE):
            return base
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = [m for m in data if m.get("role") != "system"][-20:]
            return base + data
        except Exception:
            return base

    def save_chat_history(self):
        if not CURRENT_CONFIG.get("chat_memory", True):
            return
        try:
            convo = [m for m in (self.permanent_memory + self.session_memory) if m.get("role") != "system"]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(convo[-20:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MEMORY] Save failed: {e}")

    def get_random_sound(self, directory):
        if not os.path.isdir(directory):
            return None
        files = [f for f in os.listdir(directory) if f.lower().endswith(".wav")]
        return os.path.join(directory, random.choice(files)) if files else None

    def play_sound(self, path):
        if not path:
            return
        try:
            subprocess.run(["aplay", "-q", path], timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _run_thinking_sound_loop(self):
        while self.thinking_sound_active.is_set():
            snd = self.get_random_sound(thinking_sounds_dir)
            if snd:
                self.play_sound(snd)
            else:
                time.sleep(0.3)

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
            except Exception:
                pass
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()
        if self.monitor:
            self.monitor.stop()          # releases the camera
        self.save_chat_history()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.master.quit()
        except Exception:
            pass
        threading.Timer(1.5, lambda: os._exit(0)).start()

    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget(); self.status_label.place_forget(); self.exit_button.place_forget()
                self.person_label.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
                self.person_label.place(relx=0.99, y=8, anchor=tk.NE)
        except tk.TclError:
            pass

    def handle_ptt_toggle(self, event=None):
        now = time.time()
        if now - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = now
        if self.recording_active.is_set():
            self.recording_active.clear()
        elif self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
            self.recording_active.set()
            self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING):
            self.interrupted.set()
            self.thinking_sound_active.clear()
            self.clear_tts()
            if self.current_audio_process:
                try:
                    self.current_audio_process.terminate()
                except Exception:
                    pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"]
        for state in states:
            folder = os.path.join("faces", state)
            self.animations[state] = []
            if os.path.exists(folder):
                for f in sorted(x for x in os.listdir(folder) if x.lower().endswith('.png')):
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if self.animations.get("idle"):
                    self.animations[state] = self.animations["idle"]
                else:
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation); return
        if self.current_state == BotStates.SPEAKING:
            self.current_frame_index = random.randint(1, len(frames) - 1) if len(frames) > 1 else 0
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)
        self.background_label.config(image=frames[self.current_frame_index])
        self.master.after(50 if self.current_state == BotStates.SPEAKING else 500, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg:
                print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg:
                self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except Exception:
                    pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, text + ("\n" if newline else ""))
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        self.append_to_text(chunk, newline=False)

    # =====================================================================
    # 4. ACTION ROUTER
    # =====================================================================
    def execute_action_and_get_result(self, action_data):
        raw_action = str(action_data.get("action", "")).lower().strip()
        value = action_data.get("value") or action_data.get("query")
        VALID_TOOLS = {"get_time", "search_web", "capture_image"}
        ALIASES = {"google": "search_web", "browser": "search_web", "news": "search_web",
                   "search_news": "search_web", "look": "capture_image", "see": "capture_image",
                   "check_time": "get_time"}
        action = ALIASES.get(raw_action, raw_action)
        print(f"ACTION: {raw_action} -> {action}", flush=True)

        if action not in VALID_TOOLS:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."

        if action == "search_web":
            print(f"Searching web for: {value}...", flush=True)
            try:
                with DDGS() as ddgs:
                    results = []
                    for fn in (ddgs.news, ddgs.text):
                        try:
                            results = list(fn(value, region='wt-wt', max_results=2))
                            if results:
                                break
                        except Exception as e:
                            print(f"[DEBUG] Search error: {e}", flush=True)
                    if results:
                        parts = []
                        for r in results:
                            body = r.get('body', r.get('snippet', ''))
                            parts.append(f"Title: {r.get('title', '')}\nSnippet: {body[:300]}")
                        return f"SEARCH RESULTS for '{value}':\n" + "\n---\n".join(parts)
                    return "SEARCH_EMPTY"
            except Exception as e:
                print(f"[DEBUG] Search library error: {e}", flush=True)
                return "SEARCH_ERROR"

        if action == "capture_image":
            return "IMAGE_CAPTURE_TRIGGERED"
        return None

    # =====================================================================
    # 5. CORE LOOP
    # =====================================================================
    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            threading.Thread(target=self._tts_synth_worker, daemon=True).start()
            threading.Thread(target=self._tts_play_worker, daemon=True).start()
            self.start_vitamin_monitor()
            while True:
                trigger = self.detect_wake_word_or_ptt()
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue
                self.set_state(BotStates.LISTENING, "I'm listening!")
                audio_file = self.record_voice_ptt() if trigger == "PTT" else self.record_voice_adaptive()
                if not audio_file:
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue
                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text, img_path=None)
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Checking cloud connection...")
        missing = []
        if not get_key(CURRENT_CONFIG["stt"]["key_env"]):
            missing.append(CURRENT_CONFIG["stt"]["key_env"])
        for p in CURRENT_CONFIG["llm_providers"]:
            if not get_key(p["key_env"]):
                missing.append(p["key_env"])
        if missing:
            print(f"[WARN] Missing API keys in .env: {sorted(set(missing))}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("Cloud services ready.", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()
        if self.oww_model:
            self.oww_model.reset()
        else:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE = 1280  # 80 ms at 16 kHz, what openwakeword expects
        # Prefer a native 16 kHz stream (best accuracy). Otherwise use the device rate and resample properly.
        input_rate = 16000
        try:
            sd.check_input_settings(device=INPUT_DEVICE_NAME, samplerate=16000, channels=1, dtype="int16")
        except Exception:
            input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        use_resampling = input_rate != 16000
        input_chunk = int(round(CHUNK_SIZE * input_rate / 16000)) if use_resampling else CHUNK_SIZE
        stream_args = {"samplerate": input_rate, "channels": 1, "dtype": 'int16',
                       "blocksize": input_chunk, "device": INPUT_DEVICE_NAME, "latency": "high"}
        try:
            self._listen_loop(stream_args, input_chunk, CHUNK_SIZE, use_resampling)
        except StopIteration as si:
            return str(si)
        except Exception as e:
            print(f"[AUDIO] Stream failed: {e}. Retrying with loose settings...", flush=True)
            try:
                stream_args.update({"blocksize": 1024, "latency": "high"})
                self._listen_loop(stream_args, 1024, CHUNK_SIZE, True)
            except StopIteration as si:
                return str(si)
            except Exception as e2:
                print(f"[CRITICAL] Wake Word Stream Error: {e2}", flush=True)
                self.ptt_event.wait()
                return "PTT"
        return "WAKE"

    def _listen_loop(self, stream_args, input_chunk_size, target_chunk_size, use_resampling):
        from math import gcd
        from scipy.signal import resample_poly
        in_rate = stream_args["samplerate"]
        g = gcd(16000, in_rate)
        up, down = 16000 // g, in_rate // g
        threshold = CURRENT_CONFIG.get("wake_word_threshold", WAKE_WORD_THRESHOLD)
        debug = CURRENT_CONFIG.get("wake_word_debug", True)
        peak, last_print = 0.0, 0.0
        from collections import deque
        recent_peaks = deque(maxlen=25)          # ~2 s of history for the auto-gain
        hits = deque(maxlen=4)                   # last 4 frames: wake if 2 of them were above threshold
        gain_max = float(CURRENT_CONFIG.get("wake_word_max_gain", 6.0))
        applied_gain = 1.0

        with sd.InputStream(**stream_args) as stream:
            print(f"[AUDIO] Listening at {in_rate} Hz | wake threshold {threshold}", flush=True)
            while True:
                if self.ptt_event.is_set():
                    self.ptt_event.clear()
                    raise StopIteration("PTT")
                rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                if rlist:
                    sys.stdin.readline()
                    print("[CLI] Enter pressed in terminal -> manual trigger", flush=True)
                    raise StopIteration("CLI")
                try:
                    data, overflow = stream.read(input_chunk_size)
                    if overflow:
                        print("!", end="", flush=True)
                except Exception as e:
                    raise RuntimeError(f"Audio read failed: {e}")

                audio = np.asarray(data).flatten()
                if use_resampling:
                    audio = resample_poly(audio.astype(np.float32), up, down)
                    audio = np.clip(audio, -32768, 32767).astype(np.int16)
                # openwakeword wants exactly int16 samples; pad/trim to a clean 1280 chunk
                if len(audio) != target_chunk_size:
                    audio = np.pad(audio, (0, max(0, target_chunk_size - len(audio))))[:target_chunk_size]

                raw_vol = int(np.max(np.abs(audio)))
                # Auto gain: if you are far from the mic or the mic is quiet, boost the voice
                # (openwakeword was trained on normal-loudness speech)
                if gain_max > 1.0:
                    recent_peaks.append(raw_vol)
                    ref = max(recent_peaks)
                    applied_gain = min(gain_max, max(0.25, 8000.0 / ref)) if ref > 0 else 1.0
                    if abs(applied_gain - 1.0) > 0.05:
                        audio = np.clip(audio.astype(np.float32) * applied_gain, -32768, 32767).astype(np.int16)

                self.oww_model.predict(audio)
                for mdl in self.oww_model.prediction_buffer.keys():
                    score = float(list(self.oww_model.prediction_buffer[mdl])[-1])
                    peak = max(peak, score)
                    if debug and time.time() - last_print > 0.3:
                        clip = "  MIC TOO LOUD - lower Capture volume in alsamixer" if raw_vol >= 32000 else ""
                        print(f"\r[Oww] mic {raw_vol:5d} x{applied_gain:.1f} | score {score:.2f} | peak {peak:.2f} | need {threshold}{clip}   ",
                              end="", flush=True)
                        last_print = time.time()
                    hits.append(score > threshold)
                    if score >= max(0.8, threshold) or sum(hits) >= 2:
                        print(f"\n[WAKE] '{mdl}' score {score:.2f}", flush=True)
                        self.oww_model.reset()
                        return

    def record_voice_adaptive(self, filename="input.wav"):
        """Records until you stop talking. Adapts to the room's noise level and mic volume."""
        print("Recording (Adaptive)... speak now!", flush=True)
        time.sleep(0.1)
        rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        chunk_dur = 0.05
        chunk_size = int(rate * chunk_dur)
        end_silence_chunks = int(CURRENT_CONFIG.get("end_silence_seconds", 1.5) / chunk_dur)
        max_end_chunks = int(CURRENT_CONFIG.get("end_silence_max_seconds", 2.6) / chunk_dur)
        no_speech_chunks = int(CURRENT_CONFIG.get("no_speech_seconds", 7.0) / chunk_dur)
        max_chunks = int(CURRENT_CONFIG.get("max_record_seconds", 30.0) / chunk_dur)
        buffer = []
        from collections import deque
        st = {"win": deque(maxlen=30), "n": 0, "floor": None, "speech": False, "peak": 0.0, "quiet": 0, "done": False,
              "nospeech": False, "last_print": 0.0, "vol": 0.0}

        def callback(indata, frames, time_info, status):
            vol = float(np.sqrt(np.mean(np.square(indata))))
            buffer.append(indata.copy())
            st["n"] += 1
            st["vol"] = vol
            # noise floor = quietest moment in the last ~1.5 s (can't be inflated by your own voice)
            if vol > 0:
                st["win"].append(vol)
            st["floor"] = min(st["win"]) if st["win"] else None
            floor = max(st["floor"] or 0.0, 0.0005)
            start_thr = max(0.0025, floor * 2.5)     # low enough for soft voices
            if not st["speech"]:
                if vol > start_thr:
                    st["speech"] = True
                    st["peak"] = vol
                elif st["n"] >= no_speech_chunks:
                    st["nospeech"] = True
                    st["done"] = True
                return
            st["peak"] = max(st["peak"] * 0.998, vol)
            stop_thr = max(floor * 1.8, st["peak"] * 0.08, 0.002)
            st["speech_n"] = st.get("speech_n", 0) + 1
            if vol < stop_thr:
                st["quiet"] += 1
                # Patience adapts to the speaker: if they paused for a long time earlier in this
                # sentence, wait proportionally longer (up to the max) before deciding they are done.
                needed = int(min(max_end_chunks, max(end_silence_chunks, st.get("longest_pause", 0) * 1.6)))
                if st["quiet"] >= needed and st["speech_n"] >= 14:   # at least ~0.7 s of speech
                    st["done"] = True
            else:
                if st["quiet"] >= 6:     # a real pause (>= 0.3 s) inside the sentence
                    st["longest_pause"] = max(st.get("longest_pause", 0), st["quiet"])
                st["quiet"] = 0

        try:
            with sd.InputStream(samplerate=rate, channels=1, callback=callback,
                                device=INPUT_DEVICE_NAME, blocksize=chunk_size):
                while not st["done"] and st["n"] < max_chunks and not self.interrupted.is_set():
                    sd.sleep(50)
                    if time.time() - st["last_print"] > 1.0:
                        print(f"[REC] level {st['vol']:.4f} | noise {st['floor'] or 0:.4f} | "
                              f"speech={'yes' if st['speech'] else 'no'}", flush=True)
                        st["last_print"] = time.time()
        except Exception as e:
            print(f"[AUDIO ERROR] Adaptive recording failed: {e}", flush=True)
            return None
        if st["nospeech"]:
            print("[REC] No speech detected.", flush=True)
            return None
        return self.save_audio_buffer(buffer, filename, rate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.5)
        rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        buffer = []
        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())
        try:
            sd.stop(); time.sleep(0.2)
            with sd.InputStream(samplerate=rate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set():
                    sd.sleep(50)
        except Exception as e:
            print(f"[AUDIO ERROR] PTT recording failed: {e}", flush=True)
            return None
        return self.save_audio_buffer(buffer, filename, rate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer:
            return None
        audio = np.concatenate(buffer, axis=0).flatten()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        print(f"[AUDIO] Recorded {len(audio)/samplerate:.1f}s, level {rms:.4f}", flush=True)
        if rms < 0.0008:
            print("[AUDIO] Too quiet - mic may be muted or wrong device. Skipping.", flush=True)
            return None
        self.last_rms = rms
        peak = float(np.max(np.abs(audio)))
        if 0 < peak < 0.3:   # quiet voice/mic: boost so Whisper can hear it (max 50x)
            audio = audio * min(0.7 / peak, 50.0)
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing (cloud)...", flush=True)
        try:
            t0 = time.time()
            text = cloud_transcribe(filename)
            print(f"Heard: '{text}'  [STT {time.time() - t0:.1f}s]", flush=True)
            # Whisper invents phrases like "Thank you." when it only hears silence/noise
            if text.lower().strip(" .!?,") in WHISPER_GHOSTS and getattr(self, "last_rms", 1) < 0.02:
                print("[STT] Ignored likely hallucination from near-silence.", flush=True)
                return ""
            if text and CURRENT_CONFIG.get("stt_cleanup", False):
                fixed = cleanup_transcript(text)
                if fixed != text:
                    print(f"Cleaned: '{fixed}'", flush=True)
                text = fixed
            return text
        except Exception as e:
            print(f"Transcription Error: {e}", flush=True)
            self.set_state(BotStates.ERROR, "STT failed (check internet / key)")
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        if self.monitor and self.monitor.camera.running:
            try:
                import cv2
                frame = self.monitor.camera.get_frame(max_age=3.0)
                if frame is not None:
                    cv2.imwrite(BMO_IMAGE_FILE, frame)
                    return BMO_IMAGE_FILE
            except Exception as e:
                print(f"Camera (stream) error: {e}")
        try:
            subprocess.run(["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480",
                            "-o", BMO_IMAGE_FILE], check=True)
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE).rotate(rotation, expand=True)
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # =====================================================================
    # 6. CHAT & RESPOND
    # =====================================================================
    def _say(self, text, cam_path=None):
        self.thinking_sound_active.clear()
        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=cam_path)
        self.append_to_text("BOT: ", newline=False)
        self.append_to_text(text, newline=True)
        self.speak_text(text)

    def speak_text(self, text):
        chunks, _ = split_speakable(text, final=True)
        for c in chunks:
            self.enqueue_tts(c)

    def build_messages(self, text):
        now = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
        sys_msg = {"role": "system", "content": SYSTEM_PROMPT +
                   f"\n\nCURRENT DATE AND TIME: {now}. If the user asks for the time or date, "
                   "answer directly from this in their language. Do not use a tool for it."}
        history = [m for m in self.permanent_memory if m.get("role") != "system"] + self.session_memory
        return [sys_msg] + history[-16:] + [{"role": "user", "content": text}]

    def chat_and_respond(self, text, img_path=None):
        low = text.lower()
        if any(k in low for k in ("forget everything", "reset memory", "kalimutan mo lahat", "burahin ang memorya")):
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            self.enqueue_tts("Okay. Memory wiped. Binura ko na ang lahat.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        alert = None if img_path else check_emergency(text)
        if alert:
            self._say(alert)
            self.session_memory += [{"role": "user", "content": text}, {"role": "assistant", "content": alert}]
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
            return

        if not img_path and (self.handle_mood_intent(text) or self.handle_vitamin_intent(text)):
            return

        self.set_state(BotStates.THINKING, "Thinking...", cam_path=img_path)
        if img_path:
            messages = [{"role": "system", "content": VISION_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(img_path)}}]}]
        else:
            messages = self.build_messages(text)

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        full, sentence_buf = "", ""
        t_llm, first_word = time.time(), True
        try:
            is_action = False
            for content in llm_stream(messages, vision=bool(img_path)):
                if self.interrupted.is_set():
                    break
                full += content
                if first_word:
                    first_word = False
                    print(f"[TIME] AI started answering after {time.time() - t_llm:.1f}s", flush=True)
                if not full.lstrip().startswith(("{", "```")) and not is_action:
                    pass
                else:
                    is_action = True
                    self.thinking_sound_active.clear()
                    continue
                self.thinking_sound_active.clear()
                if self.current_state != BotStates.SPEAKING:
                    self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
                    self.append_to_text("BOT: ", newline=False)
                self._stream_to_text(content)
                sentence_buf += content
                chunks, sentence_buf = split_speakable(sentence_buf)
                for c in chunks:
                    self.enqueue_tts(c)

            # flush whatever is left at the end of the reply
            if not is_action and not self.interrupted.is_set():
                chunks, sentence_buf = split_speakable(sentence_buf, final=True)
                for c in chunks:
                    self.enqueue_tts(c)

            if is_action:
                action_data = self.extract_json_from_text(full)
                if action_data:
                    result = self.execute_action_and_get_result(action_data)
                    if result and result.startswith("CHAT_FALLBACK::"):
                        chat_text = result.split("::", 1)[1]
                        self._say(chat_text, img_path)
                        self.session_memory += [{"role": "user", "content": text},
                                                {"role": "assistant", "content": chat_text}]
                    elif result == "IMAGE_CAPTURE_TRIGGERED":
                        new_img = self.capture_image()
                        if new_img:
                            self.chat_and_respond(text, img_path=new_img)
                            return
                        self._say("Hindi gumana ang camera. The camera is not working.")
                    elif result == "INVALID_ACTION":
                        self._say("Hindi ko alam kung paano gawin iyan. I am not sure how to do that.", img_path)
                    elif result == "SEARCH_EMPTY":
                        self._say("Wala akong nahanap. I couldn't find anything about that.", img_path)
                    elif result == "SEARCH_ERROR":
                        self._say("Walang internet ngayon. I cannot reach the internet right now.", img_path)
                    elif result:
                        summary_prompt = [
                            {"role": "system", "content":
                                "Summarize this result in one or two short spoken sentences. "
                                "Reply in the SAME language the user used (Tagalog, English or Taglish). "
                                "Plain text only, no markdown."},
                            {"role": "user", "content": f"RESULT: {result}\nUser Question: {text}"}]
                        self.set_state(BotStates.THINKING, "Reading...")
                        self.thinking_sound_active.set()
                        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()
                        final_text = llm_complete(summary_prompt).strip()
                        self._say(final_text, img_path)
                        self.session_memory += [{"role": "user", "content": text},
                                                {"role": "assistant", "content": final_text}]
            else:
                self.append_to_text("")
                self.session_memory += [{"role": "user", "content": text},
                                        {"role": "assistant", "content": full}]

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
        except Exception as e:
            print(f"LLM Error: {e}", flush=True)
            self.thinking_sound_active.clear()
            self.set_state(BotStates.ERROR, "Brain Freeze! (check internet / keys)")
            self.play_sound(self.get_random_sound(error_sounds_dir))

    # =====================================================================
    # 6b. VITAMIN & MOOD MONITOR  (camera + OpenCV face recognition + database)
    # =====================================================================
    def start_vitamin_monitor(self):
        vcfg = CURRENT_CONFIG.get("vitamin", {})
        if not vcfg.get("enabled", True):
            print("[VITAMIN] Disabled in config.json.", flush=True)
            return
        if VitaminMonitor is None:
            print(f"[VITAMIN] Not available ({VITAMIN_IMPORT_ERROR}). Run: pip install opencv-python-headless", flush=True)
            return
        try:
            self.monitor = VitaminMonitor(vcfg, rotation=CURRENT_CONFIG.get("camera_rotation", 0))
            self.monitor.start()
            threading.Thread(target=self._vitamin_events_loop, daemon=True).start()
        except Exception as e:
            self.monitor = None
            print(f"[VITAMIN] Monitor is OFF: {e}", flush=True)

    def _announce(self, text):
        self._say(text)
        self.wait_for_tts()
        self.set_state(BotStates.IDLE, "Waiting...")

    MOOD_COLORS = {"happy": "#2e7d32", "sad": "#c62828", "neutral": "#f9a825"}

    def _show_mood_banner(self, cur):
        def _do():
            try:
                if cur and cur.get("emotion"):
                    mood = cur["emotion"]
                    self.mood_label.config(text=f"{cur['name']}:  {mood.upper()}",
                                           bg=self.MOOD_COLORS.get(mood, "#2e2e2e"),
                                           fg="black" if mood == "neutral" else "white")
                    self.mood_label.place(relx=0.5, y=6, anchor=tk.N)
                elif cur:
                    self.mood_label.config(text=f"{cur['name']}", bg="#2e2e2e", fg="white")
                    self.mood_label.place(relx=0.5, y=6, anchor=tk.N)
                else:
                    self.mood_label.place_forget()
            except tk.TclError:
                pass
        self.master.after(0, _do)

    def _vk(self, kind):
        # In simulation mode the wording is "already been detected" instead of "already took your vitamins".
        if self.monitor and self.monitor.cfg.get("simulation_mode", True):
            return {"already": "already_detected", "locked_already": "already_detected",
                    "announce_taken": "announce_detected"}.get(kind, kind)
        return kind

    def _vitamin_events_loop(self):
        while not self.exiting:
            time.sleep(1.0)
            m = self.monitor
            if not m:
                continue
            try:
                cur = m.current_person(max_age=6.0)
                label = f"Seen: {cur['name']}" if cur else ""
                self.master.after(0, lambda t=label: self.person_var.set(t))
                self._show_mood_banner(cur)
                busy = (self.intake_busy or self.current_state != BotStates.IDLE or self.tts_pending > 0
                        or self.recording_active.is_set() or self.interrupted.is_set())
                if busy or not cur:
                    continue
                vc = m.cfg
                # First time today -> run the routine automatically (simulation)
                if (vc.get("simulation_mode", True) and vc.get("auto_start_on_detect", True)
                        and m.db.count_today(cur["pid"]) < vc["max_per_day"]
                        and m.current_person(max_age=2.0)):
                    threading.Thread(target=self.run_intake_session,
                                     args=(vc.get("routine_language", "en"), "simulation"), daemon=True).start()
                    time.sleep(1.5)
                    continue
                ev = m.check_events()
                if not ev:
                    continue
                mode = vc.get("announce_language", "both")
                langs = ["en", "tl"] if mode == "both" else [mode]
                text = " ".join(vitamin_message(self._vk(ev["type"]), l, ev["name"], ev.get("time", "")) for l in langs)
                m.mark_announced(ev)
                self._announce(text)
            except Exception as e:
                print(f"[VITAMIN] event loop error: {e}", flush=True)

    def handle_mood_intent(self, text):
        if not self.monitor or parse_mood_intent is None or not parse_mood_intent(text):
            return False
        lang = detect_lang(text)
        cur = self.monitor.current_person(max_age=10.0)
        if not cur:
            reply = vitamin_message("not_seen", lang)
        elif not cur.get("emotion"):
            reply = vitamin_message("mood_unknown", lang)
        else:
            reply = vitamin_message("mood_now_" + cur["emotion"], lang, cur["name"])
        self._say(reply)
        self.session_memory += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.wait_for_tts()
        self.set_state(BotStates.IDLE, "Ready")
        return True

    def handle_vitamin_intent(self, text):
        # Deterministic on purpose: the AI never decides whether a vitamin gets logged.
        if not self.monitor:
            return False
        kind = parse_vitamin_intent(text)
        if not kind:
            return False
        lang = detect_lang(text)
        if kind == "request":
            self.run_intake_session(lang, "voice")
            return True
        cur = self.monitor.current_person(max_age=10.0)
        if not cur:
            reply = vitamin_message("not_seen", lang)
        else:
            db, pid, name = self.monitor.db, cur["pid"], cur["name"]
            max_n = self.monitor.cfg["max_per_day"]
            if kind == "query":
                if db.count_today(pid) >= max_n:
                    reply = vitamin_message(self._vk("already"), lang, name, fmt_time(db.last_taken_today(pid) or ""))
                else:
                    reply = vitamin_message("not_yet", lang, name)
            else:
                ok, _, when = db.log_vitamin(pid, "voice", max_n)
                reply = vitamin_message("logged" if ok else self._vk("already"), lang, name, fmt_time(when or ""))
                print(f"[VITAMIN] {name}: {'LOGGED' if ok else 'REFUSED (already taken today)'}", flush=True)
        self._say(reply)
        self.session_memory += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.wait_for_tts()
        self.set_state(BotStates.IDLE, "Ready")
        return True

    def _ensure_hud(self):
        try:
            if not self.response_text.winfo_ismapped():
                self.toggle_hud_visibility()
        except tk.TclError:
            pass

    def _camera_preview(self):
        # Saves the newest camera frame so it can be shown on screen while the routine runs.
        try:
            import cv2
            frame = self.monitor.camera.get_frame(max_age=3.0)
            if frame is not None:
                cv2.imwrite(BMO_IMAGE_FILE, frame)
                return BMO_IMAGE_FILE
        except Exception:
            pass
        return None

    def run_intake_session(self, lang="en", source="voice"):
        """5 s 'take your medicine' -> 10 s mood check (Happy / Sad / Neutral) -> says the mood."""
        m = self.monitor
        if not m or self.intake_busy:
            return
        self.intake_busy = True
        self.interrupted.clear()
        try:
            cur = m.current_person(max_age=10.0)
            if not cur:
                self._say(vitamin_message("not_seen", lang))
                return
            pid, name = cur["pid"], cur["name"]
            res = m.register_intake(pid, source)          # records today's intake/detection (refuses a 2nd one)
            if res["status"] == "already":
                self._say(vitamin_message(self._vk("already"), lang, name, fmt_time(res.get("time") or "")))
                return

            vcfg = m.cfg
            prompt_s = int(vcfg.get("intake_prompt_seconds", 5))
            mood_s = int(vcfg.get("mood_check_seconds", 10))
            self.master.after(0, self._ensure_hud)

            # ---- PHASE 1: statement + camera (default 5 s) ----
            self._say(vitamin_message("take_now", lang, name))
            for left in range(prompt_s, 0, -1):
                self.set_state(BotStates.SPEAKING, f"Take your medicine now...  {left}", cam_path=self._camera_preview())
                time.sleep(1.0)

            # ---- PHASE 2: mood check (default 10 s) ----
            m.start_collecting(pid)
            for left in range(mood_s, 0, -1):
                live = m.current_person(max_age=3.0)
                live_mood = (live or {}).get("emotion") or "..."
                self.set_state(BotStates.THINKING, f"Mood check  {left}s   {str(live_mood).upper()}",
                               cam_path=self._camera_preview())
                time.sleep(1.0)
            mood, conf = m.summarize_moods(m.stop_collecting())

            # ---- RESULT: exactly one of Happy / Sad / Neutral ----
            if mood is None:
                self._say(vitamin_message("mood_unknown", lang))
            else:
                m.db.log_mood(pid, mood, conf)
                if mood == "sad":
                    m.mark_sad_announced(pid)
                self.set_state(BotStates.SPEAKING, f"Mood: {mood.upper()}", cam_path=self._camera_preview())
                self._say(vitamin_message("mood_" + mood, lang, name))
        except Exception as e:
            print(f"[INTAKE] error: {e}", flush=True)
        finally:
            try:
                m.stop_collecting()
                cur_now = m.current_person(max_age=10.0)
                if cur_now:       # we just spoke to this person: don't announce "already detected" right after
                    m.mark_announced({"type": "announce_taken", "pid": cur_now["pid"]})
            except Exception:
                pass
            self.wait_for_tts()
            self.intake_busy = False
            self.set_state(BotStates.IDLE, "Ready")

    def handle_dispense_key(self, event=None):
        if self.monitor and not self.intake_busy:
            threading.Thread(target=self.run_intake_session, args=("en", "keyboard"), daemon=True).start()

    def handle_manual_vitamin(self, event=None):
        # Press the V key (caregiver): confirm the vitamin for whoever the camera currently sees.
        if not self.monitor:
            return
        def work():
            cur = self.monitor.current_person(max_age=10.0)
            if not cur:
                self._announce(vitamin_message("not_seen", "en"))
                return
            ok, _, when = self.monitor.db.log_vitamin(cur["pid"], "keyboard", self.monitor.cfg["max_per_day"])
            self._announce(vitamin_message("logged" if ok else self._vk("already"), "en", cur["name"], fmt_time(when or "")))
        threading.Thread(target=work, daemon=True).start()

    # =====================================================================
    # 7. TEXT-TO-SPEECH PIPELINE  (synthesize next sentence while the current one plays)
    # =====================================================================
    def enqueue_tts(self, text):
        text = clean_for_tts(text)
        if not text or not re.search(r"\w", text):
            return
        with self.tts_queue_lock:
            self.tts_pending += 1
            self.tts_queue.append((self.tts_gen, text))

    def clear_tts(self):
        with self.tts_queue_lock:
            self.tts_gen += 1
            self.tts_queue.clear()
            for _, path in self.audio_queue:
                try:
                    os.remove(path)
                except Exception:
                    pass
            self.audio_queue.clear()
            self.tts_pending = 0

    def wait_for_tts(self):
        time.sleep(0.2)
        while self.tts_pending > 0 and not self.interrupted.is_set():
            time.sleep(0.1)

    def _tts_done(self, gen):
        with self.tts_queue_lock:
            if gen == self.tts_gen:
                self.tts_pending = max(0, self.tts_pending - 1)

    def _tts_synth_worker(self):
        while True:
            item = None
            with self.tts_queue_lock:
                if self.tts_queue and len(self.audio_queue) < 2:
                    item = self.tts_queue.pop(0)
            if not item:
                time.sleep(0.03)
                continue
            gen, text = item
            if gen != self.tts_gen:
                continue
            path = None
            try:
                print(f"[TTS] {text}", flush=True)
                path = cloud_tts_to_file(text)
            except Exception as e:
                print(f"[TTS ERROR] {e}", flush=True)
            queued = False
            with self.tts_queue_lock:
                if path and gen == self.tts_gen:
                    self.audio_queue.append((gen, path))
                    queued = True
            if not queued:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                self._tts_done(gen)

    def _tts_play_worker(self):
        while True:
            item = None
            with self.tts_queue_lock:
                if self.audio_queue:
                    item = self.audio_queue.pop(0)
            if not item:
                time.sleep(0.03)
                continue
            gen, path = item
            try:
                if gen == self.tts_gen:
                    player = ["aplay", "-q", path] if path.endswith(".wav") else ["mpg123", "-q", path]
                    proc = subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    self.current_audio_process = proc
                    _, err = proc.communicate()
                    if proc.returncode not in (0, -15, -9):
                        print(f"[PLAY ERROR] {player[0]} exit {proc.returncode}: "
                              f"{(err or b'').decode(errors='ignore').strip()[:300]}", flush=True)
            except Exception as e:
                print(f"[PLAY ERROR] {e}", flush=True)
            finally:
                self.current_audio_process = None
                try:
                    os.remove(path)
                except Exception:
                    pass
                self._tts_done(gen)


if __name__ == "__main__":
    root = tk.Tk()
    app = BotGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.safe_exit()
