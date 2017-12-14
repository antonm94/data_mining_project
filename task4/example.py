import numpy as np

def set_articles(articles):
    article_id = articles.keys()

     # timestamps = articles.keys()
    # user_features = np.zeros((len(timestamps), 6))
    # available_articles = np.empty(len(timestamps), object)
    # for i in range(len(timestamps)):
    #     user_features[i] = articles.get(timestamps[i])[0:5]
    #     print articles.get(timestamps[i])[6:]
    # print user_features



def update(reward):
    pass


def recommend(time, user_features, choices):
    return np.random.choice(choices)
