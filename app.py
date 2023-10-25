from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained machine learning model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define a route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the form submission and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    porepress1 = float(request.form['porepress1'])
    soiltention1 = float(request.form['soiltention1'])
    h1x = float(request.form['h1x'])
    h1y = float(request.form['h1y'])
    h1z = float(request.form['h1z'])
    inclination1y = float(request.form['inclination1y'])

  

    # Make a prediction using the loaded model
    prediction = model.predict([[porepress1,soiltention1,h1x,h1y,h1z,inclination1y]])[0]

    # Format the prediction as a text message
    prediction_text = f"Predicted inclination x = {prediction:.3f}"

    # Return the prediction message to be displayed on the HTML page
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
