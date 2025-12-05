from flask import Flask, render_template, request, jsonify
from combined_classifier.ML.main import run_classifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("user_text")
    results = run_classifier(user_text)
    return render_template("results.html", movies=results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)