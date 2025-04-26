from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    data = np.array([values])
    prediction = model.predict(data)
    return render_template('index.html', prediction_text='Recommended Crop: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
