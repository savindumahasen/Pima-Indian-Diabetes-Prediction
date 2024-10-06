from flask import Flask, render_template, request,jsonify
import pickle


app =Flask(__name__)


model_path = "./pickle.h5"
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', method=['POST'])
def predict():
    feature_names= ['0','1','2','3','4','5','6','7']
    features =[float(request.form(f)) for f in feature_names]
    input_data = [features]

    ## Make thw prediction
    prediction = model.predict(input_data)
    probabilities = model.predict(input_data)[0]

    if prediction[0]==0:
        result = "Heart Disease is not predicted"
    else:
        result = "Heart Disase predicted";


    return jsonify({'result':result, 'probebilities':probabilities[1]})



if __name__ =='__main__':
    app.run(debug=True)

  



