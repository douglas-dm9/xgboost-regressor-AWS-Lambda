import pandas as pd
import numpy as np
import bentoml 
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'toyota.csv',sep=',')
df = data.copy()

categorical_features = ['model', 'transmission', 'fuelType']
numerical_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize','price']

le = LabelEncoder()
sc = MinMaxScaler()   


df[categorical_features] = df[categorical_features].apply(le.fit_transform)

y= np.array(df.price)
X= np.array(df.drop('price',axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)

xg_reg = xgb.XGBRegressor( n_estimators = 1000,
    learning_rate=0.09, 
    min_child_weight=5,
    max_depth = 3,
    subsample = 0.75,
    seed=7)

xg_reg.fit(X_train,y_train)


def encoder(application_data):
    
    models = ['Auris', 'Avensis', 'Aygo', 'C-HR', 'Camry', 'Corolla',
           'GT86', 'Hilux', 'IQ', 'Land Cruiser', 'PROACE VERSO',
           'Prius', 'RAV4', 'Supra', 'Urban Cruiser', 'Verso',
           'Verso-S', 'Yaris']
    
    models_labels = np.arange(0,18,1)
    dict_models = dict(map(lambda i,j : (i,j) , models ,models_labels))

    transmission = ['Automatic', 'Manual', 'Other', 'Semi-Auto']
    transmission_labels = np.arange(0,5,1)
    dict_transmission = dict(map(lambda i,j : (i,j) , transmission ,transmission_labels))

    fuel_type = ['Diesel', 'Hybrid', 'Other', 'Petrol']
    fuel_type_labels =np.arange(0,5,1)
    dict_fuel_type= dict(map(lambda i,j : (i,j) , fuel_type ,fuel_type_labels))
    
    application_data['model'] = dict_models[application_data['model']]
    application_data['fuelType'] = dict_fuel_type[application_data['fuelType']]
    application_data['transmission'] = dict_transmission[application_data['transmission']]
    
    return application_data

bento_model = bentoml.xgboost.save_model("xgboost_regressor", xg_reg,
custom_objects={
        "pre_processor": encoder,
        "scaler":sc
    }
)
model_loaded = bentoml.xgboost.load_model("xgboost_regressor:latest")

#prediction = model_loaded.predict(X_test[0])
#print(f'Predicion:{prediction}')

      