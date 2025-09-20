import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from datetime import datetime

# Load .env if present
load_dotenv()

PORT = 5002

# Directory to save failed prompts for later analysis
FAILED_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "failed_prompts")
os.makedirs(FAILED_PROMPTS_DIR, exist_ok=True)

# Create Flask app
app = Flask(__name__, static_url_path="/")
CORS(app, origins=["http://localhost:3000"])

if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY not found in environment. Set it in .env or your system env.")

client = genai.Client()


@app.route("/api/process_notes", methods=["POST"])
def process_notes():
    data = request.get_json() or {}
    topic = data.get("topic", "")
    notes = data.get("notes", "")
    tone = data.get("tone", "General")

    prompt = f"""
    You are a note-taking assistant. The user provides shorthand notes that may be incomplete,
    incoherent, or contain symbols. Your job is to expand them into clear notes and return 
    structured JSON with additional extracted information.

    Tones may include: Class Lecture, Formal Meeting, Informal Meeting, Scientific Talk, 
    Business Plan, Travel Plan.

    Rules:
    1. Stay faithful to the provided notes, topic, and tone. Do not hallucinate new content.
    2. Expand shorthand into clear, plain text notes in the given tone.
    3. The "expandedNotes" field must be a single string, but preserve line breaks (`\\n`) 
       and allow simple bullet points using a hyphen (`-`).
    4. If shorthand or unclear phrases cannot be expanded, keep them as is or mark with [NEEDS CLARIFICATION].
    5. If the notes contain "??", provide a concise textbook-level explanation and record it under "explain".
    6. If notes contain multiple asterisks (** or ****), treat as important and record under "important".
    7. Identify and record tasks, reminders, or actionable items under "tasks".
    8. Identify and record meeting-related information (schedule, setup, follow-up) under "meeting".
    9. Output must be valid JSON only, no extra text, no Markdown, no explanations.

    JSON Schema (must follow exactly):
    {{
      "expandedNotes": "string (multiline, with \\n and - for bullets)",
      "important": ["list of important points"],
      "explain": ["list of misunderstood/?? terms with explanation"],
      "tasks": ["list of tasks/reminders"],
      "meeting": ["list of meeting-related info"]
    }}

    Topic: {topic}
    Tone: {tone}
    Scribbled Notes:
    {notes}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",  # enforce JSON
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        # Try to parse JSON
        try:
            parsed = json.loads(response.text)
        except Exception:
            # fallback: wrap everything in expandedNotes if JSON invalid
            parsed = {
                "expandedNotes": response.text.strip(),
                "important": [],
                "explain": [],
                "tasks": [],
                "meeting": []
            }

        return jsonify(parsed)

    except Exception as e:
        print("Error generating content:", e)

        def save_failed_prompt(prompt_text):
            try:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                filename = f"failed_prompt_{ts}.txt"
                path = os.path.join(FAILED_PROMPTS_DIR, filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                print("Saved failed prompt to:", path)
            except Exception as save_e:
                print("Failed to save prompt:", save_e)

        err_text = str(e)
        lowered = err_text.lower()
        if ("503" in err_text) or ("unavailable" in lowered) or ("model is overloaded" in lowered):
            try:
                save_failed_prompt(prompt)
            except Exception:
                pass
            return jsonify({"error": "The AI model is not available. Please try again later."}), 503

        return jsonify({"error": err_text}), 500


# Production: serve the built React app (if frontend/build exists)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    build_dir = app.static_folder  # ../frontend/build
    if path != "" and os.path.exists(os.path.join(build_dir, path)):
        return send_from_directory(build_dir, path)
    elif os.path.exists(os.path.join(build_dir, "index.html")):
        return send_from_directory(build_dir, "index.html")
    else:
        return jsonify({"status": "Backend running. Frontend build not found."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
