# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf


app = Flask(__name__)
app.secret_key = 'O.\x89\xcc\xa0>\x96\xf7\x871\xa2\xe6\x9a\xe4\x14\x91\x0e\xe5)\xd9'

# Load the Random Forest CLassifier model
filename = 'Models/diabetes-model.pkl'
filename1 = 'Models/cancer-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
rf = pickle.load(open(filename1, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('d_result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('diabetes'))

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if request.method == 'POST':
        rad = float(request.form['Radius_mean'])
        tex = float(request.form['Texture_mean'])
        par = float(request.form['Perimeter_mean'])
        area = float(request.form['Area_mean'])
        smooth = float(request.form['Smoothness_mean'])
        compact = float(request.form['Compactness_mean'])
        con = float(request.form['Concavity_mean'])
        concave = float(request.form['concave points_mean'])
        sym = float(request.form['symmetry_mean'])
        frac = float(request.form['fractal_dimension_mean'])
        rad_se = float(request.form['radius_se'])
        tex_se = float(request.form['texture_se'])
        par_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smooth_se = float(request.form['smoothness_se'])
        compact_se = float(request.form['compactness_se'])
        con_se = float(request.form['concavity_se'])
        concave_se = float(request.form['concave points_se'])
        sym_se = float(request.form['symmetry_se'])
        frac_se = float(request.form['fractal_dimension_se'])
        rad_worst = float(request.form['radius_worst'])
        tex_worst = float(request.form['texture_worst'])
        par_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smooth_worst = float(request.form['smoothness_worst'])
        compact_worst = float(request.form['compactness_worst'])
        con_worst = float(request.form['concavity_worst'])
        concave_worst = float(request.form['concave points_worst'])
        sym_worst = float(request.form['symmetry_worst'])
        frac_worst = float(request.form['fractal_dimension_worst'])

        data = np.array([[rad, tex, par, area, smooth, compact, con, concave, sym, frac, rad_se, tex_se, par_se, area_se, smooth_se, compact_se, con_se, concave_se,
                          sym_se, frac_se, rad_worst, tex_worst, par_worst, area_worst, smooth_worst, compact_worst, con_worst, concave_worst, sym_worst, frac_worst]])
        my_prediction = rf.predict(data)

        return render_template('c_result.html', prediction=my_prediction)


def ValuePredictor(to_predict_list, size):
    loaded_model = joblib.load('models/heart_model')
    to_predict = np.array(to_predict_list).reshape(1, size)
    result = loaded_model.predict(to_predict)
    return result[0]

disease_name_value = {15: 'Fungal infection',
 4: 'Allergy',
 16: 'GERD',
 9: 'Chronic cholestasis',
 14: 'Drug Reaction',
 33: 'Peptic ulcer diseae',
 1: 'AIDS',
 12: 'Diabetes ',
 17: 'Gastroenteritis',
 6: 'Bronchial Asthma',
 23: 'Hypertension ',
 30: 'Migraine',
 7: 'Cervical spondylosis',
 32: 'Paralysis (brain hemorrhage)',
 28: 'Jaundice',
 29: 'Malaria',
 8: 'Chicken pox',
 11: 'Dengue',
 37: 'Typhoid',
 40: 'hepatitis A',
 19: 'Hepatitis B',
 20: 'Hepatitis C',
 21: 'Hepatitis D',
 22: 'Hepatitis E',
 3: 'Alcoholic hepatitis',
 36: 'Tuberculosis',
 10: 'Common Cold',
 34: 'Pneumonia',
 13: 'Dimorphic hemmorhoids(piles)',
 18: 'Heart attack',
 39: 'Varicose veins',
 26: 'Hypothyroidism',
 24: 'Hyperthyroidism',
 25: 'Hypoglycemia',
 31: 'Osteoarthristis',
 5: 'Arthritis',
 0: '(vertigo) Paroymsal  Positional Vertigo',
 2: 'Acne',
 38: 'Urinary tract infection',
 35: 'Psoriasis',
 27: 'Impetigo'}

@app.route("/totalhealth", methods=['GET', 'POST'])
def totalhealth():
    return render_template('total_health.html')

@app.route("/predtotalhealth", methods=['GET', 'POST'])
def predtotalhealth():
    if request.method == 'POST':  
        f = request.files['file']  
        fpath = 'static/'+f.filename
        f.save(fpath)  
        final_features = pd.read_csv(fpath)
        print(final_features.shape)
        model_path = 'Models/total_health.pkl'
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict(final_features)
        pred = disease_name_value[result[0]]
        return render_template('t_result.html', pred=pred)

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

def predict(values, dic):
    if len(values) == 18:
        model = pickle.load(open('Models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            result = ValuePredictor(to_predict_list, 11)

            if(int(result) == 1):
                prediction = 1
            else:
                prediction = 0

            return render_template('h_result.html', prediction=prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('heart'))

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return redirect(url_for('kidney'))
    return render_template('k_result.html', pred=pred)


# this function use to predict the output for Fetal Health from given data
def fetal_health_value_predictor(data):
    try:
        # after get the data from html form then we collect the values and
        # converts into 2D numpy array for prediction
        data = list(data.values())
        data = list(map(float, data))
        data = np.array(data).reshape(1, -1)
        # load the saved pre-trained model for new prediction
        model_path = 'Models/fetal-health-model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        result = model.predict(data)
        result = int(result[0])
        status = True
        # returns the predicted output value
        return (result, status)
    except Exception as e:
        result = str(e)
        status = False
        return (result, status)


# this route for prediction of Fetal Health
@app.route('/fetal_health', methods=['GET', 'POST'])
def fetal_health_prediction():
    if request.method == 'POST':
        # geting the form data by POST method
        data = request.form.to_dict()
        # passing form data to castome predict method to get the result
        result, status = fetal_health_value_predictor(data)
        if status:
            # if prediction happens successfully status=True and then pass uotput to html page
            return render_template('fetal_health.html', result=result)
        else:
            # if any error occured during prediction then the error msg will be displayed
            return f'<h2>Error : {result}</h2>'

    # if the user send a GET request to '/fetal_health' route then we just render the html page
    # which contains a form for prediction
    return render_template('fetal_health.html', result=None)


def strokeValuePredictor(s_predict_list):
    '''function to predict the output by data we get
    from the route'''

    model = joblib.load('Models/stroke_model.pkl')
    data = np.array(s_predict_list).reshape(1, -1)
    result = model.predict(data)
    return result[0]


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/predict_stroke', methods=['POST'])
# this route for predicting chances of stroke
def predict_stroke():

    if request.method == 'POST':
        s_predict_list = request.form.to_dict()
        s_predict_list = list(s_predict_list.values())
        # list to keep the values of the dictionary items of request.form field
        s_predict_list = list(map(float, s_predict_list))
        result = strokeValuePredictor(s_predict_list)

        if(int(result) == 1):
            prediction = 1
        else:
            prediction = 0
        return render_template('st_result.html', prediction=prediction)


def liverprediction(final_features):
    # Loading the pickle file
    model_path = 'Models/liver-disease_model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    result = model.predict(final_features)
    return result[0]


@app.route('/liver')
def liver():
    return render_template('liver.html')


@app.route('/predict_liver', methods=['POST'])
# predicting
def predict_liver_disease():

    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = liverprediction(final_features)
        pred = int(output)

        return render_template('liver_result.html', prediction=pred)


@app.route("/malaria", methods=['GET', 'POST'])
def malaria():
    return render_template('malaria.html')


@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredict():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((50, 50))
                img = np.asarray(img)
                img = img.reshape((1, 50, 50, 3))
                img = img.astype(np.float64)

                model_path = "Models/malaria-model.h5"
                model = tf.keras.models.load_model(model_path)
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

@app.route("/parkinson", methods=['GET', 'POST'])
def parkinson():
    return render_template('parkinson.html')

@app.route('/predict_parkinson',methods=['POST']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        mdvp_fo=float(request.form['mdvp_fo'])
        mdvp_fhi=float(request.form['mdvp_fhi'])
        mdvp_flo=float(request.form['mdvp_flo'])
        mdvp_jitper=float(request.form['mdvp_jitper'])
        mdvp_jitabs=float(request.form['mdvp_jitabs'])
        mdvp_rap=float(request.form['mdvp_rap'])
        mdvp_ppq=float(request.form['mdvp_ppq'])
        jitter_ddp=float(request.form['jitter_ddp'])
        mdvp_shim=float(request.form['mdvp_shim'])
        mdvp_shim_db=float(request.form['mdvp_shim_db'])
        shimm_apq3=float(request.form['shimm_apq3'])
        shimm_apq5=float(request.form['shimm_apq5'])
        mdvp_apq=float(request.form['mdvp_apq'])
        shimm_dda=float(request.form['shimm_dda'])
        nhr=float(request.form['nhr'])
        hnr=float(request.form['hnr'])
        rpde=float(request.form['rpde'])
        dfa=float(request.form['dfa'])
        spread1=float(request.form['spread1'])
        spread2=float(request.form['spread2'])
        d2=float(request.form['d2'])
        ppe=float(request.form['ppe'])
            
        filename = 'modelForPrediction.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        scaler = pickle.load(open('standardScalar.sav', 'rb'))
        prediction=loaded_model.predict(scaler.transform([[mdvp_fo,mdvp_fhi,mdvp_flo,mdvp_jitper, mdvp_jitabs,
                mdvp_rap,mdvp_ppq, jitter_ddp, mdvp_shim, mdvp_shim_db,shimm_apq3,shimm_apq5,mdvp_apq,shimm_dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]]))
        print('prediction is', prediction)
        if prediction == 1:
            pred = "You have Parkinson's Disease. Please consult a specialist."
            return render_template('p_result.html', prediction=pred)
        else:
            pred = "You are Healthy Person."
            return render_template('p_result.html', prediction=pred)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=False)
