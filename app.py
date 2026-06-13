from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import joblib


model = load_model('addiction_model.h5')
scaler = joblib.load('scaleraddiction.pkl')
app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def hello_world():
    prediction = None
    if request.method == 'POST':
        # EXACT COLUMN MATCH PROTOCOL ENGAGED
        data = [
            float(request.form.get('Daily_Usage_Hours')),
            float(request.form.get('Phone_Checks_Per_Day')),
            float(request.form.get('Apps_Used_Daily')),
            float(request.form.get('Time_on_Social_Media')),
            float(request.form.get('Time_on_Gaming')),
            float(request.form.get('Sleep_Hours'))
        ]
        i=0
        while i < len(data):
            data[i]=data[i]/100
            i=i+1

        # Scale and predict
        data_scaled = scaler.transform([data])

        # Extract the raw Sigmoid probability (e.g., 0.854321...)
        raw_prediction = model.predict(data_scaled)[0][0]

        # The S-Rank Percentage Translation
        prediction = int(round(raw_prediction * 100))

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
