from flask import Flask, render_template, request
import joblib

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            
            height_input = float(request.form['height'])
            
            prediction = model.predict([[height_input]])[0]
        except ValueError:
            prediction = "Invalid input. Please enter a valid number."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5667)
