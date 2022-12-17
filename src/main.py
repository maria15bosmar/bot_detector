# Imports necesarios para el modelo
import pandas as pd
import numpy as np
import os, sys
PATH = os.path.abspath("")
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from generate_user import get_user_data


#if __name__ == "__main__":
def prob(username, num_mod):
    # Leer el argumento que sera el username de la cuenta a detectar
    # username = sys.argv[1]

    # Obtener datos del usuario.
    user_data = get_user_data(username)


    ### PARA CONSTRUIR EL MODELO

    # Leer el modelo guardado 
    xgb_ = xgb.XGBClassifier(eval_metric="mlogloss", booster= "gbtree", nthread= 1)  # init model
    xgb_.load_model(f'bot_detect_xgb_{num_mod}.model')  # load data 

    ### PARA LOS DATOS DEL USUARIO

    # Normalización de los datos del usuario.
    scaler_m = MinMaxScaler()
    user_d = np.array(user_data)
    user_array= user_d.reshape(-1, 1)
    user_array = scaler_m.fit_transform(user_array)
    final = user_array.T[0]

    # Imprime la probabilidad de que el usuario preparado antes sea bot.
    probabilidad = round(xgb_.predict_proba(np.array([final]))[0][1] * 100, 2)
    print(f"Usuario: {username}")
    print(f"Probabilidad de que sea bot: {probabilidad}%.")
    result = xgb_.predict(np.array([final]))[0]
    label_names = ["humano", "bot"]
    print(f"Clasificado como: {label_names[result]}.")
    return result

    

