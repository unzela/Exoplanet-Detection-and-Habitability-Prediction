from flask import Flask , request, render_template
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def exop():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(filename)
        df = pd.read_csv(filename, header=None)
        x = df.iloc[:,1:]
        print()
        print()
        pred = model.predict(x)
        print("hello")
        # for i in pred:
        #     if i>0.5:
        #         print("It's a Exoplanet")
        #     else:
        #         print("It's not an Exoplanet")
        for i in pred:
            try:
                if i > 0.5:
                    print("It's a ExoPlanet")
                else:
                    print("It's not a ExoPlanet")
            except:
                print("It's not a ExoPlanet")
        print()
        print()
        return render_template('index.html',pred=pred)

def preprocess(data):
    compact_data = data.drop(['P_GEO_ALBEDO', 'P_DETECTION_MASS', 'P_DETECTION_RADIUS', 'P_ALT_NAMES', 'P_ATMOSPHERE', 'S_DISC', 'S_MAGNETIC_FIELD', 
                 'P_TEMP_MEASURED', 'P_GEO_ALBEDO_ERROR_MIN', 'P_GEO_ALBEDO_ERROR_MAX', 'P_TPERI_ERROR_MAX', 'P_TPERI_ERROR_MIN', 'P_TPERI', 
                 'P_DENSITY', 'P_ESCAPE', 'P_GRAVITY', 'P_POTENTIAL', 'P_OMEGA_ERROR_MAX', 'P_OMEGA_ERROR_MIN', 'P_OMEGA', 'P_INCLINATION_ERROR_MAX', 
                 'P_INCLINATION_ERROR_MIN', 'P_INCLINATION', 'P_ECCENTRICITY_ERROR_MAX', 'P_ECCENTRICITY_ERROR_MIN', 'S_AGE_ERROR_MIN', 'S_AGE_ERROR_MAX', 
                 'P_IMPACT_PARAMETER_ERROR_MIN', 'P_IMPACT_PARAMETER_ERROR_MAX', 'P_IMPACT_PARAMETER', 'P_MASS_ERROR_MAX', 'P_MASS_ERROR_MIN', 'P_HILL_SPHERE', 
                 'P_MASS'], axis = 1)
    compact_data['S_TYPE'] = compact_data['S_TYPE'].fillna(compact_data['S_TYPE'].mode()[0])
    compact_data['P_TYPE_TEMP'] = compact_data['P_TYPE_TEMP'].fillna(compact_data['P_TYPE_TEMP'].mode()[0])
    compact_data['S_TYPE_TEMP'] = compact_data['S_TYPE_TEMP'].fillna(compact_data['S_TYPE_TEMP'].mode()[0])
    compact_data['P_TYPE'] = compact_data['P_TYPE'].fillna(compact_data['P_TYPE'].mode()[0])
    lencoders = {}
    for col in compact_data.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        compact_data[col] = lencoders[col].fit_transform(compact_data[col])

    features = compact_data[['P_TYPE_TEMP','P_PERIOD','S_DEC','S_DISTANCE','S_MASS','S_TEMPERATURE','P_TYPE','S_TIDAL_LOCK','P_HABZONE_OPT','P_RADIUS_EST']]

    # Normalize Features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    return features

@app.route('/csv/upload',methods=['GET','POST'])
def csv_upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        extension = filename.split('.')[-1]
        filename = filename.split('.')[0]+'_1'+extension
        file.save(filename)
        df = pd.read_csv(filename)
        print(df.head())
        print(df['P_HABITABLE'])
        x = preprocess(df)
        print(x)
        model_sgd = pickle.load(open('model_sgd.pickle','rb+'))
        p = model_sgd.predict(x)
        print()
        print()
        print("hello")
        # for i in pred:
        #     if i>0.5:
        #         print("It's a Exoplanet")
        #     else:
        #         print("It's not an Exoplanet")
        print(p)
        print()
        print()
        return render_template('csv.html',pred=list(zip(p,df['P_NAME'].values)))

@app.route('/csv')
def csv():
    return render_template('csv.html')

if __name__=='__main__':
    app.run(debug=True, use_reloader=True)