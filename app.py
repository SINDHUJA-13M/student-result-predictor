from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model, accuracy = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(accuracy*100,2))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        internal_marks = float(request.form["internal_marks"])

        features = np.array([[study_hours, attendance, internal_marks]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][prediction][0] * 100

        result = "Pass 🎉" if prediction[0] == 1 else "Fail ❌"

        return render_template(
            "index.html",
            prediction_text=result,
            probability=round(probability,2),
            accuracy=round(accuracy*100,2)
        )

    except:
        return render_template("index.html", error="Invalid Input!", accuracy=round(accuracy*100,2))

if __name__ == "__main__":
    app.run(debug=True)