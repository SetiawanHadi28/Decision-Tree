from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("iris_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            sepal_length = float(request.form["SepalLengthCm"])
            sepal_width = float(request.form["SepalWidthCm"])
            petal_length = float(request.form["PetalLengthCm"])
            petal_width = float(request.form["PetalWidthCm"])

            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)[0]

            return render_template("index.html", prediction=prediction)
        except Exception as e:
            return render_template("index.html", error=str(e))
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
