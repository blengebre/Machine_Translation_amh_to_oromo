import os
import requests
from flask import Flask, request, jsonify, render_template

# Import both forward (Amharic -> Oromo) and back (Oromo -> Amharic) translation functions
from inference import translate as translate_forward
from inference_back import translate as translate_back

app = Flask(__name__)

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "http://localhost:11434/v1/chat/completions"
)


def refine_translation_with_qwen(translation, direction):
    """Improve fluency and preserve the meaning of a model-generated translation."""
    if not QWEN_API_KEY or not translation.strip():
        return translation

    target_language = "Oromo" if direction == "am2om" else "Amharic"
    prompt = (
        f"You are a careful {target_language} language editor. Improve the following "
        "machine translation so it expresses the complete intended meaning naturally. "
        f"Return only the final {target_language} text, with no explanation, labels, "
        "or quotation marks. Do not add information that is not present.\n\n"
        f"Machine translation:\n{translation}"
    )

    try:
        response = requests.post(
            QWEN_API_URL,
            headers={
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Preserve meaning, names, numbers, and the requested language.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=30,
        )
        response.raise_for_status()
        refined = response.json()["choices"][0]["message"]["content"].strip()
        return refined or translation
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        app.logger.warning("Qwen refinement unavailable; returning original translation: %s", error)
        return translation

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"].strip()
    direction = data.get("direction", "am2om")  # 'am2om' or 'om2am'
    
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
    
    try:
        if direction == "om2am":
            translation = translate_back(text)
        else:
            translation = translate_forward(text)
        translation = refine_translation_with_qwen(translation, direction)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
