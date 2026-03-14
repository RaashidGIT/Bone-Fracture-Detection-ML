from flask import Flask, render_template, request
import os
from predictor import predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result, status, color = predict(filepath)

        return render_template("index.html",
                               uploaded=file.filename,
                               result=result,
                               status=status,
                               color=color)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
