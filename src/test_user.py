import nltk
import pandas as pd
import numpy as np
import os, sys
PATH = os.path.abspath("")[:-3]
from scrap import scrap_user
from sklearn.feature_extraction.text import CountVectorizer

def get_user_data(user):
    """ Devuelve los datos de un usuario listos para probar en el modelo. """
    # SCRAPPEAR UN USUARIO
    tweets = scrap_user(user)

    # CÁLCULO DEL FACTOR REP
    #tweets = pd.read_csv(PATH + "data/example.csv") # Se puede leer el csv de antes.
    contents = tweets["content"]
    toks = []
    sw = nltk.corpus.stopwords
    stemmer = nltk.stem.snowball.EnglishStemmer()
    # Tokenizar, stemmear, filtrar stop-words.
    for content in contents.values:
        tk = nltk.word_tokenize(content)
        filtered_sentence = [w for w in tk if not w.lower() in sw.words('english')]
        toks.append([stemmer.stem(word) for word in filtered_sentence])
    # Se cuenta número de repeticiones de palabras y se suman.
    try:
        vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, ngram_range=(1, 15), min_df=0.1).fit_transform(toks)
        factor_rep = vect.sum()
    except:
        factor_rep = 0

    # REPETICIÓN DE LAS FECHAS
    from datetime import datetime
    date = tweets["date"]
    hours = []
    for h in date.values:
        hours.append(pd.to_datetime(h).hour)
    valores_finales=[]
    tot= len(hours)
    result = dict()
    #contando el numero de repeticiones
    for j in hours:
        if j not in result:
            result[j] = 0 
        result[j] += 1
    #viendo el mayor valor y computando porcentaje
    values=[]
    mayor=0
    for dato, valor in result.items():
        if valor > mayor:
            mayor= valor
    hora_rep = mayor/tot

    # DATOS MEDIOS
    avg_likes = np.average(tweets["likeCount"].values)
    avg_retweets = np.average(tweets["retweetCount"].values)
    avg_reply = np.average(tweets["replyCount"].values)
    avg_quote = np.average(tweets["quoteCount"].values)

    # El usuario listo para predecir se guarda en user_data.
    user_data = tweets[["verified", "followersCount", "friendsCount", "tweetsCount", "listedCount", "mediaCount"]].iloc[-1, :].values.tolist()
    user_data.insert(0, factor_rep)
    user_data.extend([avg_reply, avg_retweets, avg_likes, avg_quote, hora_rep, user_data[2]-user_data[3]])
    return user_data