"""
Vitamin & Mood Monitor  (Raspberry Pi 5 + Camera Module 3 + OpenCV)

What it does
  * Watches the camera and recognises WHO is in front of it (face recognition, runs 100% on the Pi).
  * Keeps a small SQLite database: people, vitamin intake per day, and mood (Happy / Sad / Neutral).
  * If a person already took their vitamins today it tells them, and refuses to log a second one.
  * Checks mood (Happy / Sad / Neutral) from the face and logs it for the caregiver.

Privacy: NO photos or video are saved. Only a 128-number "face fingerprint" per sample is stored in
vitamins.db on this Pi. Nothing is sent to the cloud. Get the person's (or their guardian's) consent.
This is a reminder helper, NOT a medical device. Doses are decided by the doctor.

Command line (stop agent.py first - the camera can only be used by one program):
  python vitamin_monitor.py enroll "Lola Rosa"     register a person (8 quick face samples)
  python vitamin_monitor.py test                   live test: who / mood / FPS, saves test_view.jpg
  python vitamin_monitor.py list                   show people
  python vitamin_monitor.py report 7               last 7 days of vitamins + mood
  python vitamin_monitor.py mark "Lola Rosa"       caregiver: log today's vitamin manually
  python vitamin_monitor.py undo "Lola Rosa"       remove today's entry (use this to RESET the simulation for that person)
  python vitamin_monitor.py delete "Lola Rosa"     delete a person and ALL their data
"""
import os
import re
import sys
import time
import atexit
import sqlite3
import argparse
import datetime
import threading
import subprocess
import shutil
from collections import deque, Counter

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
FER_FILE = "facial_expression_recognition_mobilefacenet_2022july.onnx"

# The FER model outputs 7 classes in this order. We only keep happy / neutral / sad.
FER_CLASSES = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
MOODS = {"happy": 3, "neutral": 4, "sad": 5}

DEFAULTS = {
    "enabled": True,
    "db_path": "vitamins.db",
    "max_per_day": 1,                  # how many vitamin intakes are allowed per day
    "match_threshold": 0.40,           # face similarity needed to say "this is Lola Rosa" (higher = stricter)
    "match_margin": 0.05,              # best person must beat the 2nd best by this much
    "detect_score": 0.80,              # face detector confidence
    "min_face_px": 70,                 # ignore faces smaller than this (too far away to recognise)
    "detect_every_seconds": 0.7,
    "announce_cooldown_seconds": 10,   # minimum seconds between two "already detected" announcements
    "announce_language": "both",       # "en", "tl" or "both"
    "sad_check_enabled": True,
    "sad_cooldown_seconds": 3600,
    "sad_min_confidence": 0.60,        # below this, a "sad" reading is called neutral (fewer false alarms)
    # --- SIMULATION MODE (no servo / no real medicine box yet)
    "simulation_mode": True,           # being detected counts as the "intake": 1st time -> routine, later -> "already detected"
    "auto_start_on_detect": True,      # start the routine automatically the first time a person is seen today
    "reappear_gap_seconds": 8,         # person must be out of view this long to count as "seen again"
    "routine_language": "en",          # language of the routine: "en" or "tl"
    # --- intake session: "take your medicine" prompt, then mood check
    "intake_prompt_seconds": 5,
    "mood_check_seconds": 10,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 10,
}


def merged(cfg):
    d = dict(DEFAULTS)
    d.update(cfg or {})
    return d


def fmt_time(iso):
    try:
        return datetime.datetime.fromisoformat(iso).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return ""


# =========================================================================
# 1. MESSAGES + VOICE-COMMAND PARSING  (pure functions, easy to test)
# =========================================================================
MESSAGES = {
    "not_seen": {
        "en": "I can't see you yet. Please look at the camera.",
        "tl": "Hindi pa po kita nakikita. Pakiharap po sa camera.",
    },
    "already": {
        "en": "{name}, you already took your vitamins today at {t}. Please do not take another one. "
              "If you are not sure, ask your doctor.",
        "tl": "{name}, nakainom ka na po ng vitamins ngayong araw, noong {t}. Huwag na po uling uminom. "
              "Kung hindi sigurado, magtanong po sa doktor.",
    },
    "not_yet": {
        "en": "{name}, you have not taken your vitamins today. Take them the way your doctor told you.",
        "tl": "{name}, hindi pa po kayo nakakainom ng vitamins ngayong araw. Inumin po ayon sa bilin ng doktor.",
    },
    "logged": {
        "en": "Okay {name}! I wrote down your vitamins at {t}. Good job!",
        "tl": "Sige po, {name}! Naitala ko na ang vitamins mo, {t}. Galing!",
    },
    "take_now": {
        "en": "{name}, please take your medicine now.",
        "tl": "{name}, inumin na po ang gamot ninyo ngayon.",
    },
    "mood_happy": {
        "en": "Mood check: Happy. You look happy!",
        "tl": "Mood check: Happy. Mukha kang masaya po!",
    },
    "mood_sad": {
        "en": "Mood check: Sad. You look sad. Are you okay?",
        "tl": "Mood check: Sad. Mukha kang malungkot po. Okay ka lang po ba?",
    },
    "mood_neutral": {
        "en": "Mood check: Neutral. You look calm.",
        "tl": "Mood check: Neutral. Mukha kang kalmado po.",
    },
    "mood_unknown": {
        "en": "I could not see your face for the mood check.",
        "tl": "Hindi ko po nakita ang mukha ninyo para sa mood check.",
    },
    "announce_detected": {
        "en": "{name}, you have already been detected.",
        "tl": "{name}, na-detect ka na po.",
    },
    "already_detected": {
        "en": "{name}, you have already been detected today at {t}.",
        "tl": "{name}, na-detect ka na po kanina, noong {t}.",
    },
    "mood_now_happy": {
        "en": "{name}, your mood is Happy.",
        "tl": "{name}, Happy po ang mood ninyo. Masaya po kayo!",
    },
    "mood_now_sad": {
        "en": "{name}, your mood is Sad. Are you okay?",
        "tl": "{name}, Sad po ang mood ninyo. Okay lang po ba kayo?",
    },
    "mood_now_neutral": {
        "en": "{name}, your mood is Neutral.",
        "tl": "{name}, Neutral po ang mood ninyo. Kalmado po kayo.",
    },
    "announce_taken": {
        "en": "{name}, you already took your vitamins.",
        "tl": "{name}, nakainom ka na po ng vitamins mo.",
    },
    "sad_check": {
        "en": "{name}, you look a little sad. Are you okay?",
        "tl": "{name}, parang malungkot ka po. Okay ka lang po ba?",
    },
}


def vitamin_message(kind, lang="en", name="", t=""):
    lang = lang if lang in ("en", "tl") else "en"
    return MESSAGES[kind][lang].format(name=name, t=t).replace("  ", " ")


_VIT = re.compile(r"vitamin|bitamina", re.I)
_QUERY = re.compile(
    r"(did|have) i( already| even)? (take|taken|took)|"
    r"(can|may|should) i (take|have|drink) (my |the )?vitamin|"
    r"nakainom na ba|nainom ko na ba|nainom na ba|naka-?take na ba|"
    r"pwede na ba|puwede na ba|dapat na ba", re.I)
_TAKEN = re.compile(
    r"\bi (just |already |have |'ve )?(took|taken|drank)\b|\bi('ve| have) (just )?(taken|drunk)|"
    r"uminom na (po )?(ako|ko)|nainom ko na|nakainom na (po )?(ako|ko)|ininom ko na|nag-?take na (po )?(ako|ko)",
    re.I)


_REQUEST = re.compile(
    r"(give|bring|dispense|get|open) (me )?(my |the )?(vitamin|medicine|box)|"
    r"i (want|need|would like|'d like|will|'ll|am going to|wanna) (to )?(take|have|drink|get)( my| the)? vitamin|"
    r"i want my vitamin|time (for|to take) (my )?vitamin|take my vitamins? now|vitamins? please|"
    r"\biinom\b|\biinumin\b|painom|pakibigay|pakiabot|bigyan (mo )?(po )?(ako|ko|kami)|ibigay (mo )?(po )?(ang )?(vitamin|bitamina)|"
    r"oras na (ng|para sa) (ang )?(vitamin|bitamina)|gusto (ko|kong|ko pong) (uminom|inumin|mag-?take|kumuha)", re.I)


_MOOD_Q = re.compile(
    r"\bmy mood\b|how do i look|am i (happy|sad)|what('s| is) my mood|mood ko|ano (ang )?mood|anong mood|"
    r"masaya ba ako|malungkot ba ako|ano ang itsura ko", re.I)


def parse_mood_intent(text):
    """True when the person asks about their own mood (answer comes from the camera)."""
    return bool(_MOOD_Q.search(text))


def parse_vitamin_intent(text):
    """Returns 'query' (asking if they already took it), 'taken' (saying they took it),
    'request' (asking for their vitamins now) or None."""
    if not _VIT.search(text):
        return None
    if _QUERY.search(text):
        return "query"
    if _TAKEN.search(text):
        return "taken"
    if _REQUEST.search(text):
        return "request"
    return None


# =========================================================================
# 2. DATABASE
# =========================================================================
class VitaminDB:
    def __init__(self, path):
        self.path = path if os.path.isabs(path) else os.path.join(HERE, path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS people(
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS face_samples(
                id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS vitamin_log(
                id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                taken_at TEXT NOT NULL, day TEXT NOT NULL, source TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS mood_log(
                id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                ts TEXT NOT NULL, day TEXT NOT NULL, emotion TEXT NOT NULL, confidence REAL);
        """)
        self.conn.commit()

    @staticmethod
    def today():
        return datetime.date.today().isoformat()

    @staticmethod
    def now_iso():
        return datetime.datetime.now().isoformat(timespec="seconds")

    def person_id(self, name):
        with self.lock:
            r = self.conn.execute("SELECT id FROM people WHERE name=? COLLATE NOCASE", (name,)).fetchone()
            return r[0] if r else None

    def add_person(self, name):
        with self.lock:
            pid = self.person_id(name)
            if pid:
                return pid
            cur = self.conn.execute("INSERT INTO people(name, created_at) VALUES(?,?)", (name, self.now_iso()))
            self.conn.commit()
            return cur.lastrowid

    def add_sample(self, pid, emb):
        emb = np.asarray(emb, dtype=np.float32).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        with self.lock:
            self.conn.execute("INSERT INTO face_samples(person_id, embedding) VALUES(?,?)", (pid, emb.tobytes()))
            self.conn.commit()

    def load_people(self):
        people = []
        with self.lock:
            for pid, name in self.conn.execute("SELECT id, name FROM people ORDER BY id").fetchall():
                rows = self.conn.execute("SELECT embedding FROM face_samples WHERE person_id=?", (pid,)).fetchall()
                if not rows:
                    continue
                samples = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
                people.append({"id": pid, "name": name, "samples": samples})
        return people

    def count_today(self, pid):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM vitamin_log WHERE person_id=? AND day=?",
                                     (pid, self.today())).fetchone()[0]

    def last_taken_today(self, pid):
        with self.lock:
            r = self.conn.execute("SELECT taken_at FROM vitamin_log WHERE person_id=? AND day=? "
                                  "ORDER BY id DESC LIMIT 1", (pid, self.today())).fetchone()
            return r[0] if r else None

    def log_vitamin(self, pid, source, max_per_day):
        """Atomic. Returns (logged, count_today, time_iso). Refuses when the daily limit is reached."""
        with self.lock:
            count = self.count_today(pid)
            if count >= max_per_day:
                return False, count, self.last_taken_today(pid)
            now = self.now_iso()
            self.conn.execute("INSERT INTO vitamin_log(person_id, taken_at, day, source) VALUES(?,?,?,?)",
                              (pid, now, self.today(), source))
            self.conn.commit()
            return True, count + 1, now

    def undo_last_today(self, pid):
        with self.lock:
            r = self.conn.execute("SELECT id FROM vitamin_log WHERE person_id=? AND day=? ORDER BY id DESC LIMIT 1",
                                  (pid, self.today())).fetchone()
            if not r:
                return False
            self.conn.execute("DELETE FROM vitamin_log WHERE id=?", (r[0],))
            self.conn.commit()
            return True

    def log_mood(self, pid, emotion, conf):
        with self.lock:
            self.conn.execute("INSERT INTO mood_log(person_id, ts, day, emotion, confidence) VALUES(?,?,?,?,?)",
                              (pid, self.now_iso(), self.today(), emotion, float(conf)))
            self.conn.commit()

    def list_people(self):
        with self.lock:
            return self.conn.execute(
                "SELECT p.name, p.created_at, COUNT(f.id) FROM people p LEFT JOIN face_samples f "
                "ON f.person_id=p.id GROUP BY p.id ORDER BY p.id").fetchall()

    def delete_person(self, name):
        with self.lock:
            pid = self.person_id(name)
            if not pid:
                return False
            for tbl in ("face_samples", "vitamin_log", "mood_log"):
                self.conn.execute(f"DELETE FROM {tbl} WHERE person_id=?", (pid,))
            self.conn.execute("DELETE FROM people WHERE id=?", (pid,))
            self.conn.commit()
            return True

    def report(self, days):
        since = (datetime.date.today() - datetime.timedelta(days=days - 1)).isoformat()
        out = []
        with self.lock:
            for pid, name in self.conn.execute("SELECT id, name FROM people ORDER BY id").fetchall():
                vit = self.conn.execute("SELECT day, taken_at, source FROM vitamin_log WHERE person_id=? AND day>=? "
                                        "ORDER BY id", (pid, since)).fetchall()
                moods = self.conn.execute("SELECT day, emotion, COUNT(*) FROM mood_log WHERE person_id=? AND day>=? "
                                          "GROUP BY day, emotion ORDER BY day", (pid, since)).fetchall()
                out.append((name, vit, moods))
        return out


# =========================================================================
# 3. CAMERA  (rpicam-vid MJPEG pipe: works with Camera Module 3 + autofocus, no extra Python packages)
# =========================================================================
def split_jpegs(buf):
    """Pulls complete JPEG frames out of a byte buffer. Returns (frames, leftover)."""
    frames = []
    while True:
        a = buf.find(b"\xff\xd8")
        if a < 0:
            return frames, buf[-1:] if buf else buf
        b = buf.find(b"\xff\xd9", a + 2)
        if b < 0:
            return frames, buf[a:]
        frames.append(buf[a:b + 2])
        buf = buf[b + 2:]


class CameraStream:
    def __init__(self, width=640, height=480, fps=10, rotation=0):
        self.width, self.height, self.fps, self.rotation = width, height, fps, rotation
        self.proc = None
        self.frame = None
        self.frame_time = 0.0
        self.running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self, wait=8.0):
        binary = shutil.which("rpicam-vid") or shutil.which("libcamera-vid")
        if not binary:
            raise RuntimeError("rpicam-vid not found. Install: sudo apt install rpicam-apps")
        cmd = [binary, "-t", "0", "-n", "--codec", "mjpeg", "--width", str(self.width),
               "--height", str(self.height), "--framerate", str(self.fps),
               "--autofocus-mode", "continuous", "--flush", "-o", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.running = True
        atexit.register(self.stop)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        t0 = time.time()
        while time.time() - t0 < wait:
            if self.frame is not None:
                return
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode(errors="ignore")[-400:]
                self.running = False
                raise RuntimeError("Camera failed to start (is another program using it? stop agent.py / "
                                   f"rpicam-hello first). Details: {err.strip()}")
            time.sleep(0.1)
        self.stop()
        raise RuntimeError("Camera produced no frames. Test it with: rpicam-hello -t 5000")

    def _reader(self):
        buf = b""
        try:
            while self.running and self.proc and self.proc.poll() is None:
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                frames, buf = split_jpegs(buf)
                if len(buf) > 4_000_000:
                    buf = b""
                if frames:
                    img = cv2.imdecode(np.frombuffer(frames[-1], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if self.rotation == 90:
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        elif self.rotation == 180:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        elif self.rotation == 270:
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        with self._lock:
                            self.frame, self.frame_time = img, time.time()
        except Exception as e:
            print(f"[CAMERA] reader stopped: {e}", flush=True)
        self.running = False

    def get_frame(self, max_age=2.0):
        with self._lock:
            if self.frame is None or time.time() - self.frame_time > max_age:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


# =========================================================================
# 4. FACE ENGINE: detect (YuNet) + recognise (SFace) + mood (FER)  - all OpenCV, all local
# =========================================================================
class FaceEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        paths = {k: os.path.join(MODEL_DIR, v) for k, v in
                 (("yunet", YUNET_FILE), ("sface", SFACE_FILE), ("fer", FER_FILE))}
        missing = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing model files {missing}. Run:  bash download_models.sh")
        mk_det = getattr(cv2, "FaceDetectorYN_create", None) or cv2.FaceDetectorYN.create
        mk_rec = getattr(cv2, "FaceRecognizerSF_create", None) or cv2.FaceRecognizerSF.create
        self.detector = mk_det(paths["yunet"], "", (320, 320), float(cfg["detect_score"]), 0.3, 5000)
        self.recognizer = mk_rec(paths["sface"], "")
        self.fer = cv2.dnn.readNet(paths["fer"])

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        faces = [f for f in faces if f[2] >= self.cfg["min_face_px"]]
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def embed(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)      # 112x112, same template the FER model expects
        emb = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9), aligned

    def mood(self, aligned):
        """Returns ('happy'|'neutral'|'sad', confidence 0-1)."""
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        blob = cv2.dnn.blobFromImage(rgb)
        try:
            self.fer.setInput(blob, "data")
            out = self.fer.forward("label")
        except cv2.error:
            self.fer.setInput(blob)
            out = self.fer.forward()
        logits = np.array([out.flatten()[i] for i in MOODS.values()], dtype=np.float64)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        names = list(MOODS.keys())
        k = int(np.argmax(p))
        label, conf = names[k], float(p[k])
        if label == "sad" and conf < float(self.cfg["sad_min_confidence"]):
            label, conf = "neutral", float(p[names.index("neutral")])
        return label, conf


# =========================================================================
# 5. THE MONITOR (background thread used by agent.py)
# =========================================================================
class VitaminMonitor:
    def __init__(self, cfg, rotation=0):
        self.cfg = merged(cfg)
        self.db = VitaminDB(self.cfg["db_path"])
        self.engine = FaceEngine(self.cfg)
        self.camera = CameraStream(self.cfg["camera_width"], self.cfg["camera_height"],
                                   self.cfg["camera_fps"], rotation)
        self._collect = None            # list of (mood, confidence) while a mood check is running
        self._collect_pid = None
        self.people = self.db.load_people()
        self._current = None
        self._lock = threading.Lock()
        self._streak = (None, 0)
        self._moods = {}
        self._last_emotion_t = 0.0
        self._last_mood_log = {}
        self._announced = {}
        self._episodes = {}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.people:
            print("[VITAMIN] No one is enrolled yet. Run:  python vitamin_monitor.py enroll \"Name\"", flush=True)
        self.camera.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[VITAMIN] Monitor running. People enrolled: {[p['name'] for p in self.people]}", flush=True)

    def stop(self):
        self._stop.set()
        self.camera.stop()

    def reload_people(self):
        self.people = self.db.load_people()

    # ---- matching ----
    def match(self, emb):
        """Returns (person_dict or None, best_similarity)."""
        scores = []
        for p in self.people:
            scores.append((float(np.max(p["samples"] @ emb)), p))
        if not scores:
            return None, 0.0
        scores.sort(key=lambda s: s[0], reverse=True)
        best, person = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if best >= self.cfg["match_threshold"] and (best - second) >= self.cfg["match_margin"]:
            return person, best
        return None, best

    # ---- background loop ----
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            frame = self.camera.get_frame(max_age=2.0)
            if frame is None:
                time.sleep(0.3)
                continue
            try:
                faces = self.engine.detect(frame)
                if not faces or not self.people:
                    self._streak = (None, 0)
                else:
                    emb, aligned = self.engine.embed(frame, faces[0])
                    person, sim = self.match(emb)
                    self._update(person, aligned)
            except Exception as e:
                print(f"[VITAMIN] loop error: {e}", flush=True)
            self._stop.wait(max(0.05, self.cfg["detect_every_seconds"] - (time.time() - t0)))

    def _update(self, person, aligned):
        if person is None:
            self._streak = (None, 0)
            return
        pid = person["id"]
        n = self._streak[1] + 1 if self._streak[0] == pid else 1
        self._streak = (pid, n)
        if n < 2:                      # must be seen twice in a row: avoids one-frame mistakes
            return
        now = time.time()
        ep = self._episodes.get(pid)
        if ep is None or now - ep["last"] > self.cfg["reappear_gap_seconds"]:
            ep = {"announced": False}                  # a new "appearance" in front of the camera
            self._episodes[pid] = ep
        ep["last"] = now
        label, conf = None, 0.0
        cur = self._current or {}
        interval = 0.6 if self._collect is not None else 1.5
        if now - self._last_emotion_t >= interval or cur.get("pid") != pid:
            label, conf = self.engine.mood(aligned)
            self._last_emotion_t = now
            if self._collect is not None and self._collect_pid == pid:
                self._collect.append((label, conf))
            self._moods.setdefault(pid, deque(maxlen=8)).append(label)
            if now - self._last_mood_log.get(pid, 0) >= 60:
                self.db.log_mood(pid, label, conf)
                self._last_mood_log[pid] = now
        moods = self._moods.get(pid)
        smoothed = Counter(list(moods)[-5:]).most_common(1)[0][0] if moods else None
        with self._lock:
            self._current = {"pid": pid, "name": person["name"], "seen": now,
                             "emotion": smoothed, "conf": conf or cur.get("conf", 0.0)}

    # ---- API for agent.py ----
    def current_person(self, max_age=8.0):
        with self._lock:
            if self._current and time.time() - self._current["seen"] <= max_age:
                return dict(self._current)
        return None

    def check_events(self):
        """Returns an event to announce, or None. Call mark_announced(ev) after speaking it."""
        cur = self.current_person(max_age=4.0)
        if not cur:
            return None
        pid, now = cur["pid"], time.time()
        ep = self._episodes.get(pid)
        if (self.db.count_today(pid) >= self.cfg["max_per_day"] and ep and not ep["announced"]
                and now - self._announced.get(("taken", pid), 0) > self.cfg["announce_cooldown_seconds"]):
            return {"type": "announce_taken", "pid": pid, "name": cur["name"],
                    "time": fmt_time(self.db.last_taken_today(pid) or "")}
        if self.cfg["sad_check_enabled"]:
            recent = list(self._moods.get(pid, []))[-8:]
            if (len(recent) >= 5 and sum(1 for m in recent if m == "sad") >= 0.6 * len(recent)
                    and now - self._announced.get(("sad", pid), 0) > self.cfg["sad_cooldown_seconds"]):
                return {"type": "sad_check", "pid": pid, "name": cur["name"], "time": ""}
        return None

    # ---- intake record + mood check ----
    def register_intake(self, pid, source="simulation"):
        """Records today's intake/detection FIRST. Returns {'status': 'registered'|'already', 'time': iso}."""
        ok, count, when = self.db.log_vitamin(pid, source, self.cfg["max_per_day"])
        return {"status": "registered" if ok else "already", "time": when}

    def start_collecting(self, pid):
        self._collect_pid = pid
        self._collect = []

    def stop_collecting(self):
        data, self._collect = self._collect or [], None
        return data

    @staticmethod
    def summarize_moods(readings):
        """readings: [(mood, conf)]. Returns (mood, avg_conf) using the most common one, or (None, 0)."""
        if not readings:
            return None, 0.0
        top = Counter(m for m, _ in readings).most_common(1)[0][0]
        confs = [c for m, c in readings if m == top]
        return top, float(sum(confs) / len(confs))

    def mark_sad_announced(self, pid):
        self._announced[("sad", pid)] = time.time()

    def mark_announced(self, ev):
        kind = "taken" if ev["type"] == "announce_taken" else "sad"
        self._announced[(kind, ev["pid"])] = time.time()
        if kind == "taken" and self._episodes.get(ev["pid"]):
            self._episodes[ev["pid"]]["announced"] = True


# =========================================================================
# 6. COMMAND LINE TOOLS
# =========================================================================
def _open_engine_and_camera(rotation=0):
    cfg = merged({})
    engine = FaceEngine(cfg)
    cam = CameraStream(cfg["camera_width"], cfg["camera_height"], cfg["camera_fps"], rotation)
    cam.start()
    return cfg, engine, cam


def cmd_enroll(args):
    name = args.name.strip()
    print(f"\nYou are about to register '{name}' for face recognition.")
    print("- Only a numeric 'face fingerprint' is stored on this device. No photos or video are saved.")
    print("- It is used only to know who is taking vitamins and to check mood. You can delete it any time.")
    if input("Does this person (or their guardian) agree? Type YES to continue: ").strip().upper() != "YES":
        print("Cancelled.")
        return
    cfg, engine, cam = _open_engine_and_camera()
    db = VitaminDB(cfg["db_path"])
    pid = db.add_person(name)
    prompts = ["Look straight at the camera", "Look straight and smile", "Turn your head a little to the left",
               "Turn your head a little to the right", "Chin up a little", "Chin down a little",
               "Look straight, normal face", "Look straight and smile again"]
    got = 0
    try:
        for prompt in prompts:
            print(f"\n[{got + 1}/{len(prompts)}] {prompt} ...", flush=True)
            time.sleep(2.5)
            for _ in range(8):
                frame = cam.get_frame()
                faces = engine.detect(frame) if frame is not None else []
                if len(faces) == 1:
                    emb, _ = engine.embed(frame, faces[0])
                    db.add_sample(pid, emb)
                    got += 1
                    print("   captured", flush=True)
                    break
                print("   need exactly ONE face, close to the camera (about 50 cm)", flush=True)
                time.sleep(0.6)
    finally:
        cam.stop()
    print(f"\nDone. Saved {got} face samples for '{name}'." +
          ("" if got >= 5 else "  (Few samples - run enroll again for better accuracy.)"))


def cmd_test(args):
    cfg, engine, cam = _open_engine_and_camera()
    db = VitaminDB(cfg["db_path"])
    people = db.load_people()
    mon_like = type("M", (), {})()
    mon_like.people, mon_like.cfg = people, cfg
    print(f"Testing for {args.seconds}s. Enrolled: {[p['name'] for p in people]}. Press Ctrl+C to stop.")
    t_end, frames, last_img = time.time() + args.seconds, 0, None
    try:
        while time.time() < t_end:
            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            frames += 1
            faces = engine.detect(frame)
            for f in faces[:1]:
                emb, aligned = engine.embed(frame, f)
                best_name, best_sim = "UNKNOWN", 0.0
                for p in people:
                    s = float(np.max(p["samples"] @ emb))
                    if s > best_sim:
                        best_sim, best_name = s, p["name"]
                if best_sim < cfg["match_threshold"]:
                    best_name = "UNKNOWN"
                mood, conf = engine.mood(aligned)
                x, y, w, h = [int(v) for v in f[:4]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_name} {mood}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                print(f"face {w}px | {best_name:12s} sim={best_sim:.2f} | mood={mood} {conf:.2f}", flush=True)
            last_img = frame
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
    if last_img is not None:
        out = os.path.join(HERE, "test_view.jpg")
        cv2.imwrite(out, last_img)
        print(f"Saved last view to {out}")
    print("Camera frames received:", frames)


def cmd_list(args):
    db = VitaminDB(DEFAULTS["db_path"])
    rows = db.list_people()
    if not rows:
        print("No one enrolled yet.")
    for name, created, n in rows:
        print(f"{name:20s} samples={n}  since {created[:10]}")


def cmd_report(args):
    db = VitaminDB(DEFAULTS["db_path"])
    for name, vit, moods in db.report(args.days):
        print(f"\n=== {name} (last {args.days} days) ===")
        print("Vitamins:")
        for day, at, src in vit:
            print(f"  {day}  {fmt_time(at)}  ({src})")
        if not vit:
            print("  none logged")
        print("Mood readings (count per day):")
        by_day = {}
        for day, emo, n in moods:
            by_day.setdefault(day, []).append(f"{emo}:{n}")
        for day, items in by_day.items():
            print(f"  {day}  " + "  ".join(items))
        if not by_day:
            print("  none")


def cmd_mark(args):
    db = VitaminDB(DEFAULTS["db_path"])
    pid = db.person_id(args.name)
    if not pid:
        print("Unknown person.")
        return
    ok, count, when = db.log_vitamin(pid, "caregiver", DEFAULTS["max_per_day"])
    print("Logged." if ok else f"Not logged: already took today at {fmt_time(when)}.")


def cmd_undo(args):
    db = VitaminDB(DEFAULTS["db_path"])
    pid = db.person_id(args.name)
    print("Removed today's last entry." if pid and db.undo_last_today(pid) else "Nothing to remove.")


def cmd_delete(args):
    db = VitaminDB(DEFAULTS["db_path"])
    if input(f"Delete '{args.name}' and ALL their data? Type YES: ").strip().upper() != "YES":
        print("Cancelled.")
        return
    print("Deleted." if db.delete_person(args.name) else "Unknown person.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vitamin & mood monitor tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("enroll"); s.add_argument("name"); s.set_defaults(fn=cmd_enroll)
    s = sub.add_parser("test"); s.add_argument("--seconds", type=int, default=30); s.set_defaults(fn=cmd_test)
    s = sub.add_parser("list"); s.set_defaults(fn=cmd_list)
    s = sub.add_parser("report"); s.add_argument("days", nargs="?", type=int, default=7); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("mark"); s.add_argument("name"); s.set_defaults(fn=cmd_mark)
    s = sub.add_parser("undo"); s.add_argument("name"); s.set_defaults(fn=cmd_undo)
    s = sub.add_parser("delete"); s.add_argument("name"); s.set_defaults(fn=cmd_delete)
    a = ap.parse_args()
    a.fn(a)
