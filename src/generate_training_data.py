""" Crea un fichero preparado para entrenar con las características que elegimos."""
import nltk
import pandas as pd
import numpy as np
import os, sys
from sklearn.feature_extraction.text import CountVectorizer
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

vers = [False, True]
# Variables a tener en cuenta.
cols = ['username', 'factor_rep', 'verified', 'followersCount', 'friendsCount',
       'tweetsCount', 'listedCount', 'mediaCount', 'avg_reply', 'avg_retweet',
       'like_reply', 'avg_quote', 'hora_rep', 'followers_diff', 'clase']
# Se leen los datos.
data = pd.read_csv("data/datos_balanced_en.csv")
training = pd.DataFrame([], columns=cols)
# Para cada usuario diferente del fichero...
users = data["username"].unique()
for ind, user in enumerate(users):
    user_data = data.loc[data["username"]==user]
    # FACTOR REP --> Text mining y agregación.
    contents = user_data["content"]
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
    #print(factor_rep)

    # REPETICIÓN DE LAS FECHAS
    date = user_data["date"]
    hours = []
    for h in date.values:
        hours.append(int(h[-15:-12]))
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
    avg_likes = np.average(user_data["likeCount"].values)
    avg_retweets = np.average(user_data["retweetCount"].values)
    avg_reply = np.average(user_data["replyCount"].values)
    avg_quote = np.average(user_data["quoteCount"].values)
    
    # Se genera por cada usuario una lista con las características elegidas y se escribe en un CSV.
    user_data = user_data[["username", "verified", "followersCount", "friendsCount", "tweetsCount", "listedCount", "mediaCount", "clase"]].iloc[-1, :].values.tolist()
    clase = user_data.pop()
    user_data[1] = vers.index(user_data[1])
    user_data.insert(1, factor_rep)
    user_data.extend([avg_reply, avg_retweets, avg_likes, avg_quote, hora_rep, user_data[2]-user_data[3], clase])
    training = pd.concat([training, pd.DataFrame([user_data], columns=cols)])
    training.to_csv("data/training_new.csv", index=False)
    print(ind)
