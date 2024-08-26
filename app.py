from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    moisture= request.form['moisture']
    soil_temp = request.form["soil temp"]

    feature_list = [N, P, K, temp, humidity, ph, moisture, soil_temp]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)

    crop_dict = {'rice': "rice", 'sugarcane': "sugarcane", 'mustard': "mustard", 'potato': "potato", 'tomato': "tomato",
                 'maize': "maize"}
    fert_dict = {'urea': "urea", 'TSP': "TSP", 'MOP': "MOP"}


    crop_prediction_value = prediction[0][0]
    fert_prediction_value = prediction[0][1]

    print("check: "+prediction[0][1]);
    crop_result = None
    fert_result = None


    if crop_prediction_value in crop_dict:
        crop = crop_dict[crop_prediction_value]
        crop_result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        crop_result = "Sorry, we could not determine the best crop to be cultivated with the provided data."


    print("prediction value"+fert_prediction_value)
    if fert_prediction_value in fert_dict:
        fert = fert_dict[fert_prediction_value]
        fert_result = "{} is the best fertilizer to be used right there".format(fert)
    else:
        fert_result = "Sorry, we could not determine the best fertilizer to be used with the provided data."
    return render_template('index.html', crop_result=crop_result, fert_result=fert_result)




# python main
if __name__ == "__main__":
    app.run(debug=True)
