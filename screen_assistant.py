# screen_assistant.py
import json
import google.generativeai as genai
import whisper
import sounddevice as sd
import numpy as np
import pyautogui
import pyttsx3
import pytesseract
import cv2
from PIL import Image
import os
import time
import subprocess
import psutil
from dotenv import load_dotenv
import urllib.parse
import re
import threading

# Optional window focus support
try:
    import pygetwindow as gw
except Exception:
    gw = None  # fallback

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_KEY)

# Choose Whisper model: use small for speed (default). Set to True to attempt loading large.
USE_WHISPER_LARGE = False

whisper_model_name = "large" if USE_WHISPER_LARGE else "medium"
print("Loading Whisper model:", whisper_model_name)
whisper_model = whisper.load_model(whisper_model_name, device="cpu")

# Behavior toggles
use_gemini = True
learning = True             # log mistakes to mistakes.json
post_verify = True          # verify an action's effect after executing

mistake_file = "mistakes.json"
if not os.path.exists(mistake_file):
    json.dump([], open(mistake_file, "w"))

# TTS engine initialization factory (to avoid driver issues when repeated)
def speak_async(text):
    if not text:
        return
    def _speak(t):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(t)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_speak, args=(text,), daemon=True).start()

def speak(text):
    print("AI:", text)
    speak_async(text)

# -----------------------------
# LISTEN + TRANSCRIBE (VAD-like)
# -----------------------------
def listen_once(samplerate=16000, silence_threshold=0.01, min_volume=0.02, silence_duration=1.0):
    print("Listening... (speak now)")
    buffer, silence_start, started = [], None, False
    try:
        with sd.InputStream(channels=1, samplerate=samplerate) as stream:
            while True:
                indata, _ = stream.read(1024)
                volume = np.sqrt(np.mean(indata ** 2))
                if volume > min_volume:
                    started = True
                if started:
                    buffer.append(indata.copy())
                    if volume < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_duration:
                            break
                    else:
                        silence_start = None
                time.sleep(0.03)
    except Exception as e:
        print("Microphone error:", e)
        return None
    if not buffer:
        return None
    audio = np.concatenate(buffer, axis=0)
    return audio

def transcribe_audio(audio):
    if audio is None or len(audio) == 0:
        return ""
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    # Whisper accepts file paths or numpy arrays depending on wrapper; using .transcribe(audio) works in your prior code.
    # Warning: large model will be slow on CPU.
    result = whisper_model.transcribe(audio, fp16=False)
    text = result.get("text", "").strip()
    print("You said:", text)
    return text

# -----------------------------
# SCREEN ANALYSIS
# -----------------------------
def analyze_screen(show_overlay=False):
    """
    Captures the screen, performs multi-pass OCR with preprocessing,
    detects button-like elements, and optionally shows attention overlay.
    Returns list of recognized UI elements with text and bounding boxes.
    """
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    # --- multiple OCR passes for better accuracy ---
    modes = [
        gray,
        cv2.bitwise_not(gray),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 15, 8)
    ]
    elements, texts = [], set()

    for mode in modes:
        data = pytesseract.image_to_data(mode, output_type=pytesseract.Output.DICT)
        for i, txt in enumerate(data['text']):
            txt = txt.strip()
            if not txt or txt.lower() in texts:
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if w < 10 or h < 10:
                continue
            elem = {'type': 'text', 'text': txt, 'position': [x, y, x + w, y + h]}
            elements.append(elem)
            texts.add(txt.lower())

    # --- detect button-like shapes (for Gemini hints) ---
    for e in elements:
        x1, y1, x2, y2 = e['position']
        w, h = x2 - x1, y2 - y1
        e['is_button_like'] = 40 < h < 150 and 120 < w < 400

    # --- show visual attention map (debugging overlay) ---
    if show_overlay:
        debug_img = img.copy()
        for e in elements:
            x1, y1, x2, y2 = e['position']
            color = (0, 255, 0) if e['is_button_like'] else (255, 0, 0)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, e['text'][:25], (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Visual Attention Map", debug_img)
        cv2.waitKey(800) 
        cv2.destroyAllWindows()

    return elements

# -----------------------------
# WINDOW MANAGEMENT (focus app/window)
# -----------------------------
def list_open_windows():
    """Return list of window titles."""
    titles = []
    if gw:
        for w in gw.getAllTitles():
            if w and w.strip():
                titles.append(w)
    else:
        # fallback: use psutil process names (less accurate)
        for p in psutil.process_iter(['name']):
            try:
                titles.append(p.info['name'])
            except Exception:
                pass
    return titles

def focus_window_by_name(name):
    name_l = name.lower().replace('.', '').strip()
    if gw:
        wins = gw.getAllWindows()
        best = None
        for w in wins:
            title = (w.title or "").lower().replace('.', '')
            if not title:
                continue
            if name_l in title or title in name_l:
                best = w
                break
        if best:
            try:
                best.activate()
                time.sleep(0.4)
                return True
            except Exception:
                try:
                    best.minimize(); best.maximize(); best.restore()
                    return True
                except Exception:
                    return False
    else:
        pyautogui.press("win")
        time.sleep(0.2)
        pyautogui.typewrite(name)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(1)
        return True
    return False

# -----------------------------
# LOCAL INTENT PARSER (fast)
# -----------------------------
def local_intent_parser(q):
    q = q.strip().lower()
    # simple patterns (add more as needed)
    m = re.match(r'open (.+)', q)
    if m:
        return {"action":"open", "target": m.group(1).strip(), "target_position": None, "content":"", "response": f"Opening {m.group(1).strip()}"}
    m = re.match(r'close (.+)', q)
    if m:
        return {"action":"close", "target": m.group(1).strip(), "target_position": None, "content":"", "response": f"Closing {m.group(1).strip()}"}
    m = re.match(r'type (.+)', q)
    if m:
        return {"action":"type", "target":"", "target_position": None, "content": m.group(1).strip(), "response": f"Typing {m.group(1).strip()}"}
    m = re.match(r'scroll (down|up)', q)
    if m:
        dir = m.group(1)
        return {"action":"scroll", "target":dir, "target_position": None, "content":"", "response": f"Scrolling {dir}"}
    m = re.match(r'search (for )?(.+)', q)
    if m:
        return {"action":"search","target":"","target_position": None, "content": m.group(2).strip(), "response": f"Searching for {m.group(2).strip()}"}
    # compute simple math like "compute 2 + 5" or "what is 2+5"
    m = re.search(r'([\d\.\s\+\-\*\/\(\)]+)$', q)
    if m and any(ch in m.group(1) for ch in "+-*/"):
        expr = m.group(1)
        return {"action":"compute", "target":"", "target_position": None, "content": expr.strip(), "response": f"Computing {expr.strip()}"}
    return None

# -----------------------------
# GEMINI PROMPT (improved)
# -----------------------------
def get_intent_gemini(user_query, screen_elements, open_windows, active_window, context):
    """
    Provide Gemini an enriched system prompt including:
      - strict JSON output format
      - active window and list of open windows
      - screen_elements (OCR)
      - rules for mapping commands to actions
    """
    system_prompt = f"""
You are a desktop automation assistant. Return STRICTLY valid JSON (no extra text).
Fields: action, target, target_position, content, response

Inputs:
- user_query: {json.dumps(user_query)}
- active_window: {json.dumps(active_window)}
- open_windows: {json.dumps(open_windows[:20])}
- screen_elements: {json.dumps(screen_elements[:80])}
  (Note: each element has 'type' field = 'text' or 'visual'. Visual means non-text items like videos or images.)

Action domain:
- action must be one of: "click","type","search","compute","open","close","scroll","generic".
- If user asked to open a website (contains .com/.org/...): action=open and target=URL.
- If user asked to open or switch to an app/window: action=open and target=app/window name.
- If user references "the previous/that window" map it to active_window or most recent.
- For clicking: prefer returning target_position = [center_x, center_y] of the OCR element or UI element on screen.
- If you cannot locate a UI element, return action=generic with a brief response explaining why.

Respond with JSON only, example:
{{
 "action":"click",
 "target":"Play",
 "target_position":[123,456],
 "content":"",
 "response":"Clicking Play for you."
}}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Build a short chat history (last few turns)
        chat_log = []
        for c in context[-6:]:
            role = c.get("role","user")
            content = c.get("content","")
            chat_log.append(f"{role.capitalize()}: {content}")
        chat_log.append(f"User: {user_query}")
        response = model.generate_content([system_prompt, "\n".join(chat_log)])
        cleaned = response.text.strip().strip("`").replace("json","").strip()
        # ensure it's JSON
        return json.loads(cleaned)
    except Exception as e:
        print("Gemini error:", e)
        return None

# -----------------------------
# MISTAKE LEARNING
# -----------------------------
def load_mistakes():
    try:
        return json.load(open(mistake_file, "r"))
    except Exception:
        return []

def save_mistake(intent):
    if not learning or not intent:
        return
    mistakes = load_mistakes()
    mistakes.append(intent)
    json.dump(mistakes[-30:], open(mistake_file, "w"), indent=2)
    print("Saved mistake.")

def avoid_repeated(intent):
    mistakes = load_mistakes()
    for m in mistakes:
        if m.get("target") == intent.get("target") or m.get("content") == intent.get("content"):
            return True
    return False

# -----------------------------
# POST-ACTION VERIFICATION
# -----------------------------
def verify_action_effect(intent, prev_screen, post_screen, prev_windows, post_windows):
    """Simple heuristics:
       - If action=open: check that target now appears in window list or URL opened.
       - If action=click and target was an element, verify that element changed/disappeared.
    """
    action = intent.get("action")
    target = (intent.get("target") or "").lower()
    if action == "open":
        # check windows
        for w in post_windows:
            if target and target in w.lower():
                return True
        # if URL, can't always see window list — assume success
        if any(ext in target for ext in [".com", ".org", ".net", ".in"]):
            return True
        return False
    if action == "click" and target:
        # if target text was present before but no longer present -> success
        prev_texts = {e['text'].lower() for e in prev_screen}
        post_texts = {e['text'].lower() for e in post_screen}
        if target in prev_texts and target not in post_texts:
            return True
        # else we can't be sure -> return False to let assistant ask user
        return False
    # for scroll, type, compute — basic success
    if action in ("scroll","type","compute","search"):
        return True
    return False

# -----------------------------
# EXECUTE ACTION
# -----------------------------
def execute_action(intent, screen_elements=None):
    if not intent:
        return
    if learning and avoid_repeated(intent):
        speak("I will skip that — you told me not to do it earlier.")
        return

    action = intent.get("action")
    target = (intent.get("target") or "").strip()
    pos = intent.get("target_position")
    content = intent.get("content") or ""

    prev_screen = screen_elements or analyze_screen()
    prev_windows = list_open_windows()

    # OPEN (app or URL) — try to focus window first if it already exists
    if action == "open" and target:
        target = target.lower().strip().replace('.', '')
        # special case: handle "edge" existing window
        if "edge" in target:
            if focus_window_by_name("microsoft edge"):
                speak("Switched to Edge.")
                return

        if any(ext in target for ext in [".com", ".org", ".net", ".in"]):
            url = target if target.startswith("http") else f"https://{target}"
            subprocess.Popen(f'start {url}', shell=True)
            speak(intent.get("response", f"Opening {target}"))
        else:
            focused = False
            if focus_window_by_name(target):
                speak(intent.get("response", f"Switched to {target}"))
                focused = True
            else:
                pyautogui.press("win")
                time.sleep(0.2)
                pyautogui.typewrite(target)
                time.sleep(0.6)
                pyautogui.press("enter")
                speak(intent.get("response", f"Opening {target}"))
                focused = True
            if not focused:
                subprocess.Popen(["start", "", target], shell=True)
        if post_verify:
            time.sleep(0.9)
            post_windows = list_open_windows()
            post_screen = analyze_screen()
            ok = verify_action_effect(intent, prev_screen, post_screen, prev_windows, post_windows)
            if not ok:
                speak(f"I couldn't confirm that {target} opened. Did it open for you?")
                save_mistake(intent)
        return

    # CLICK — if pos missing, try to find in OCR elements
    if action == "click":
        if not pos and screen_elements and target:
            for e in screen_elements:
                if target.lower() in e['text'].lower():
                    x1, y1, x2, y2 = e['position']
                    pos = [ (x1+x2)//2, (y1+y2)//2 ]
                    break
        if pos:
            pyautogui.moveTo(int(pos[0]), int(pos[1]), duration=0.15)
            pyautogui.click()
            speak(intent.get("response", "Clicked."))
        else:
            speak(intent.get("response", "I couldn't find that item on screen."))
            save_mistake(intent)
        if post_verify:
            time.sleep(0.6)
            post_screen = analyze_screen()
            ok = verify_action_effect(intent, prev_screen, post_screen, prev_windows, list_open_windows())
            if not ok:
                speak("Action didn't produce the expected change.")
                save_mistake(intent)
        return

    # TYPE
    if action == "type":
        pyautogui.typewrite(content, interval=0.03)
        pyautogui.press("enter")
        speak(intent.get("response", "Typed input."))
        return

    # SEARCH
    if action == "search":
        query = urllib.parse.quote_plus(content)
        subprocess.Popen(f'start https://www.google.com/search?q={query}', shell=True)
        speak(intent.get("response", "Search opened in browser."))
        return

    # SCROLL
    if action == "scroll":
        # simple mapping: "down" vs "up" in target
        direction = (target or "").lower()
        if "up" in direction:
            pyautogui.scroll(600)
        else:
            pyautogui.scroll(-600)
        speak(intent.get("response", "Scrolled."))
        return

    # COMPUTE
    if action == "compute":
        try:
            result = eval(content)
            speak(f"The answer is {result}")
        except Exception:
            speak("I couldn't compute that expression.")
        return

    # CLOSE
    if action == "close" and target:
        found = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if target.lower() in (proc.info['name'] or "").lower():
                    proc.terminate()
                    found = True
                    speak(intent.get("response", f"Closed {target}"))
                    break
            except Exception:
                pass
        if not found:
            speak(f"I couldn't find {target} running.")
            save_mistake(intent)
        return

    # GENERIC fallback
    speak(intent.get("response", "Okay."))

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    context = []
    speak("System ready. Hybrid mode online. Say 'exit' to quit.")
    while True:
        audio = listen_once()
        query = transcribe_audio(audio).lower().strip()
        if not query:
            continue

        if any(word in query for word in ["exit", "quit", "stop", "bye"]):
            speak("Goodbye.")
            break

        # correction/undo
        if query in ("no", "undo", "not that"):
            if learning and context:
                last = context[-1].get("intent")
                if last:
                    save_mistake(last)
                    speak("Okay — noted. I will avoid that.")
            continue

        # pipeline: reuse last detection when not needed
        # decide if we need fresh vision: if query references screen elements or click/press/scroll or 'video', take screenshot
        vision_needed = any(tok in query for tok in ("click","press","open","close","video","tab","scroll","play","pause","search","select","click on","click the"))
        screen_elements = analyze_screen()

        # local parser first
        local = local_intent_parser(query)
        intent = None
        if local:
            intent = local
        elif use_gemini:
            open_windows = list_open_windows()
            active_window = None
            try:
                if gw:
                    aw = gw.getActiveWindow()
                    active_window = aw.title if aw else ""
                else:
                    active_window = ""
            except Exception:
                active_window = ""
            intent = get_intent_gemini(query, screen_elements, open_windows, active_window, context)

        if intent:
            # store intent in context for possible undo learning
            context.append({"role":"user","content":query})
            context.append({"role":"assistant","content":intent.get("response",""), "intent":intent})
            execute_action(intent, screen_elements)
        else:
            speak("I didn't understand that. Could you rephrase?")
