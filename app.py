import os
from flask import Flask, request, jsonify, render_template

# Import both forward (Amharic -> Oromo) and back (Oromo -> Amharic) translation functions
from inference import translate as translate_forward
from inference_back import translate as translate_back

app = Flask(__name__)

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
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
