# Screen Assistant – Your On-Screen AI Controller

**Screen Assistant** is an intelligent, voice-driven desktop automation system built with **Python**, **OpenAI Whisper**, and **Google Gemini**. It enables hands-free computer control through natural voice commands — opening apps, switching windows, clicking buttons, performing searches, and analyzing the live screen using OCR and AI reasoning.

---

## 🎯 Overview

Screen Assistant listens to your voice, transcribes it via Whisper, analyzes the visible screen, and sends both the query and OCR results to Gemini for reasoning. Based on the intent, it executes actions like:

* Opening or switching applications
* Clicking UI elements
* Searching the web
* Typing text
* Performing basic computations
* Learning from user feedback for correction

This makes it a **hybrid multimodal assistant** that can both *see* and *act* intelligently.

---

## 🚀 Features

### 🧠 AI-Powered Command Understanding

* Uses **Google Gemini 2.5 Flash** for contextual action planning
* Learns from mistakes and adapts over time
* Enforces strict JSON-based intent format for safety

### 🎤 Speech Interaction

* **Whisper (medium/large)** model for high-accuracy voice-to-text
* **Text-to-Speech** via `pyttsx3` for natural spoken feedback
* Handles silence detection and voice activity dynamically

### 🖥️ Vision & Screen Analysis

* Multi-pass **OCR** via `pytesseract`
* Detects and classifies *button-like elements*
* Optional **visual overlay** showing detected text boxes and clickable regions
* Real-time screen context for Gemini reasoning

### ⚙️ Automation & Control

* Open, close, or switch between applications
* Click on-screen buttons
* Scroll, type, and perform search queries
* Compute math expressions verbally
* Verifies post-action results using OCR difference

### 🧩 Learning Mode

* Logs failed actions in `mistakes.json`
* Avoids repeating known mistakes
* Improves automation accuracy over sessions

### 💬 Example Voice Commands

| Command                | Action                           |
| ---------------------- | -------------------------------- |
| “Open WhatsApp”        | Launches or switches to WhatsApp |
| “Click on Get Started” | Locates and clicks the button    |
| “Search for AI news”   | Opens Google with query          |
| “Compute 2 plus 7”     | Calculates and speaks result     |
| “Scroll down”          | Scrolls the screen               |
| “Close Chrome”         | Terminates process               |

---

## 🧩 Project Structure

```
ScreenAssistant/
│
├── screen_assistant.py         # Main program
├── mistakes.json               # Adaptive learning log
├── .env
└── README.md
```

---

## ⚙️ Installation

### **Prerequisites**

* Python 3.9+
* Working microphone
* Windows or macOS (Windows recommended for GUI automation)
* Google Gemini API Key

### **Setup Steps**

```bash
# 1. Clone the repository
git clone https://github.com/themeetshah/screen-assistant.git
cd screen-assistant

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # on macOS/Linux
.venv\Scripts\activate         # on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 5. Run the assistant
python screen_assistant.py
```

---

## 🧠 How It Works

1. **Listening Phase** – Microphone captures user speech until silence is detected.
2. **Transcription Phase** – Whisper converts speech to text.
3. **Vision Phase** – Captures current screen and extracts text elements using OCR.
4. **Reasoning Phase** – Gemini receives the transcribed query, active windows, and OCR results, returning a JSON intent.
5. **Action Phase** – Executes the action (open, click, type, etc.) using `pyautogui`.
6. **Verification Phase** – Checks post-action state to confirm success.
7. **Learning Phase** – Logs incorrect actions for future avoidance.

---

## 🔜 Planned Enhancements

* 🧩 Object detection for non-text UI elements
* 🗣️ Real-time conversational mode (continuous listening)
* 📚 Plugin API for custom automation commands
* 🧬 Local LLM fallback for offline mode

---

## 📌 Technologies Used

| Category           | Tools                  |
| ------------------ | ---------------------- |
| Programming        | Python                 |
| Speech Recognition | OpenAI Whisper         |
| Vision             | OpenCV, Tesseract OCR  |
| Reasoning          | Google Gemini API      |
| Automation         | PyAutoGUI, PyGetWindow |
| Audio              | Pyttsx3, SoundDevice   |
| Data               | JSON, dotenv           |

---

## ⚠️ Notes

* Requires **Tesseract OCR** installed and added to system PATH.
* Gemini API key must have **generative model access** enabled.
* Some GUI automation may behave differently depending on display scaling or virtual desktops.

---

## 🧩 Example Workflow

```bash
AI: System ready. Hybrid mode online. Say 'exit' to quit.
Listening... (speak now)
You said: Open WhatsApp
AI: Opening WhatsApp.
AI: WhatsApp opened.
```


## 🤝 Contribute to CityShield

Contributions are welcomed! Feel free to contribute by creating [**pull requests**](https://github.com/themeetshah/screen-assistant/pulls) or [submitting issues](https://github.com/themeetshah/screen-assistant/issues).

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🧑‍💻 Author

**Meet Shah**  
Computer Engineering | AI + Vision + Automation Enthusiast