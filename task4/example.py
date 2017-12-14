import numpy as np
import factorUCB
d=5
index = {}


def set_articles(articles):
    global index
    # init collection of matrix/vector Aa, Ba, ba
    i = 0
    art_len = len(articles)
    article_features = np.zeros((art_len, 1, d))
    for key in articles:
        index[key] = i
        article_features[i] = articles[key][1:]
        i += 1

    factorUCB.FactorUCBArticleStruct()

def update(reward):
    pass


def recommend(time, user_features, choices):
    return np.random.choice(choices)
