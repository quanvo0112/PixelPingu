from flask import Flask, render_template, request, jsonify
from judge import score_penguin_submission

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit_artwork", methods=["POST"])
def submit_artwork():
    data = request.json
    canvas_data = data.get("canvas_data")

    if canvas_data:
        judge_results = score_penguin_submission(canvas_data)
        judge_score = judge_results.get("score", 0)
        return jsonify(
            {
                "success": True,
                "judge_score": round(judge_score, 2),
                "flag_part": judge_results.get("flag_part", ""),
            }
        )
    return jsonify({"success": False, "error": "No canvas data provided"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=25001)
