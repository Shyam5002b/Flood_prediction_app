from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the saved components
scaler = joblib.load("transform.save")
model = joblib.load("flood.save")
columns = joblib.load("columns.save") 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input values
            values = [float(request.form.get(col, 0)) for col in columns]
            input_array = np.array(values).reshape(1, -1)

            # Scale + predict
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            # Choose page based on prediction
            if prediction == 1:
                return render_template("chance.html")      # FLOOD
            else:
                return render_template("nochance.html")    # NO FLOOD

        except Exception as e:
            return f"<h3>Error: {e}</h3>"

    return render_template("home.html", columns=columns)

if __name__ == "__main__":
    app.run(debug=True)
