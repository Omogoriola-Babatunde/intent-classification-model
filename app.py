import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name)

# Load the pre-trained text classification model
model = joblib.load('intent_classification_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    # Get the text from the request data
    data = request.get_json()
    text = data.get('text')

    # Use your pre-trained model to classify the text
    result = model.predict([text])[0]

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
