import os
from flask import Flask, request, jsonify, render_template

# Import the translate function from the back translation inference script
from inference_back import translate

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index_back.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    oromo_text = data["text"].strip()
    if not oromo_text:
        return jsonify({"error": "Empty text provided"}), 400
    
    try:
        # Call the existing translate function
        amharic_translation = translate(oromo_text)
        return jsonify({"translation": amharic_translation})
    except Exception as e:
        # Return a generic error to the frontend if something goes wrong
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
