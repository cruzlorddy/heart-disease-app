
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('heart_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame({
        'Age': [data['age']],
        'Sex': [data['sex']],
        'ChestPainType': [data['chest_pain']],
        'RestingBP': [data['resting_bp']],
        'Cholesterol': [data['cholesterol']],
        'FastingBS': [data['fasting_bs']],
        'RestingECG': [data['resting_ecg']],
        'MaxHR': [data['max_hr']],
        'ExerciseAngina': [data['exercise_angina']],
        'Oldpeak': [data['oldpeak']],
        'ST_Slope': [data['st_slope']]
    })

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return jsonify({
        'prediction': prediction,
        'probability': probability
    })

if __name__ == '__main__':
    app.run(debug=True)
