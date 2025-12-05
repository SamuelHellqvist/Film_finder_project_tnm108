from flask import Blueprint, request, render_template
from ML.main import run_classifier

main_routes = Blueprint("main_routes", __name__)

@main_routes.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("user_text")
    results = run_classifier(user_text)
    return render_template("results.html", movies=results)
