"""Voice tuner + speaker test - works WITHOUT the full agent.
  python test_voice.py            -> plays the sample at 0, 2, 3.5, 5 and 7 semitones so you can compare
  python test_voice.py 4          -> plays only +4 semitones
Pick the one you like and put the number in config.json  ->  "pitch_shift_semitones"
Needs:  sudo apt install ffmpeg mpg123"""
import asyncio, os, shutil, subprocess, sys, tempfile
import edge_tts

VOICE = "fil-PH-BlessicaNeural"
TEXT = "Hello! Ako si BMO. Masakit ba ang ulo mo? Magpahinga ka muna ha!"
EDGE_PITCH, VOLUME = "+15Hz", "+25%"

for tool in ("ffmpeg", "mpg123"):
    if not shutil.which(tool):
        sys.exit(f"{tool} is missing. Run: sudo apt install {tool}")

def play(path):
    r = subprocess.run(["mpg123", "-q", path], capture_output=True, text=True)
    if r.returncode != 0:
        print("   PLAYBACK ERROR:", r.stderr.strip() or r.returncode)

levels = [float(sys.argv[1])] if len(sys.argv) > 1 else [0, 2, 3.5, 5, 7]
fd, mp3 = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
asyncio.run(edge_tts.Communicate(TEXT, VOICE, pitch=EDGE_PITCH, volume=VOLUME).save(mp3))
print("Downloaded voice OK:", os.path.getsize(mp3), "bytes")
for st in levels:
    print(f"--> +{st} semitones" + ("  (original voice)" if st == 0 else ""))
    if st == 0:
        play(mp3); continue
    out = mp3[:-4] + f"_{st}.mp3"
    f = 2 ** (st / 12)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-af",
                        f"asetrate={int(24000 * f)},aresample=24000,atempo={1 / f:.5f}", "-ar", "24000",
                        "-codec:a", "libmp3lame", "-q:a", "4", out], capture_output=True, text=True)
    if r.returncode != 0:
        print("   ffmpeg error:", r.stderr.strip()[:300]); continue
    play(out); os.remove(out)
os.remove(mp3)
