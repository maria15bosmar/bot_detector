# Imports necesarios para el modelo
import pandas as pd
import numpy as np
import os, sys
PATH = os.path.abspath("")
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from generate_user import get_user_data


if __name__ == "__main__":
    # Leer el argumento que sera el username de la cuenta a detectar
    username = sys.argv[1]

    # Obtener datos del usuario.
    user_data = get_user_data(username)


    ### PARA CONSTRUIR EL MODELO
    '''
    # Abrir base de datos de entrenamiento.
    training = pd.read_csv(PATH + os.path.sep + os.path.join("data","training_new.csv")).sample(frac=1).reset_index(drop=True)
    training["clase"].replace("bot", 1, inplace= True)
    training["clase"].replace("human", 0, inplace= True)
    training.pop("username")
    label = training.pop("clase")
    # Normalización.
    scaler = MinMaxScaler()
    data = scaler.fit_transform(training)

    # Creación del modelo con cross validation.
    skf = StratifiedKFold(n_splits = 10)
    for train_index, test_index in skf.split(data, label):
        x_train= data[train_index]
        y_train = label[train_index]
        x_test= data[test_index]
        y_test = label[test_index]
        xgb_= xgb.XGBClassifier(eval_metric="mlogloss", booster= "gbtree", nthread= 1).fit(x_train,y_train)
    '''

    # Leer el modelo guardado 
    xgb_ = xgb.XGBClassifier(eval_metric="mlogloss", booster= "gbtree", nthread= 1)  # init model
    xgb_.load_model('bot_detect_xgb.model')  # load data

    ### PARA LOS DATOS DEL USUARIO

    # Normalización de los datos del usuario.
    scaler_m = MinMaxScaler()
    user_d = np.array(user_data)
    user_array= user_d.reshape(-1, 1)
    user_array = scaler_m.fit_transform(user_array)
    final = user_array.T[0]

    # Imprime la probabilidad de que el usuario preparado antes sea bot.
    probabilidad = round(xgb_.predict_proba(np.array([final]))[0][1] * 100, 2)
    print(f"Probabilidad de que sea bot: {probabilidad}%.")
    result = xgb_.predict(np.array([final]))[0]
    label_names = ["humano", "bot"]
    print(f"Clasificado como: {label_names[result]}.")

    

