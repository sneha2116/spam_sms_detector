from flask import Flask, request, render_template
import pickle

# Load the model and vectorizer
with open('/mnt/data/spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('/mnt/data/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Create Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)

        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
