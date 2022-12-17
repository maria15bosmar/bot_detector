# Imports necesarios para el modelo
import pandas as pd
import numpy as np
import os, sys
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from generate_user import get_user_data
PATH = os.path.abspath("")


if __name__ == "__main__":

    ### PARA CONSTRUIR EL MODELO

    # Abrir base de datos de entrenamiento.
    training = pd.read_csv(PATH + os.path.sep + os.path.join("data","training_new.csv")).sample(frac=1).reset_index(drop=True)
    training["clase"].replace("bot", 1, inplace= True)
    training["clase"].replace("human", 0, inplace= True)
    training.pop("username")
    training.pop("tweetsCount")
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
        xgb_= xgb.XGBClassifier(eval_metric = "mlogloss", booster = "gbtree", nthread = 1).fit(x_train,y_train)
    xgb_.save_model("bot_detect_xgb.model")



