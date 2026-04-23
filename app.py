from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    vals=[float(request.form[k]) for k in ['area','bedrooms','bathrooms','age']]
    pred=model.predict([vals])[0]
    return render_template('index.html', prediction_text=f'Estimated Price: ₹{pred:,.0f}')
if __name__=='__main__':
    app.run(debug=True)
