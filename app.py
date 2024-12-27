from flask import Flask, request, jsonify
import model

app = Flask(__name__)

@app.route('/')
def home():
    return "Health Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")
    prediction = model.predict_health(data)
    print(f"Prediction: {prediction}")
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)