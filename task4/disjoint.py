#!/usr/bin/env python2.7

import numpy as np
from numpy import linalg
import time

method = 'disjointUCB'

class disjointUCB:
    def __init__(self):
        self.article_features = {}

        # upper bound coefficient
        self.alpha = 2.9 #1 + np.sqrt(np.log(2/delta)/2)
        r1 = 20
        r0 = -0.5
        self.r = (r0, r1)
        # dimension of user features = d
        self.d = 6
        # dimension of article features = k
        self.k = self.d*self.d
        # A0 : matrix to compute hybrid part, k*k
        self.A0 = np.identity(self.k)
        self.A0I = np.identity(self.k)
        # b0 : vector to compute hybrid part, k
        self.b0 = np.zeros((self.k, 1))
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = {}
        # AaI : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}
        # Ba : collection of matrix to compute hybrid part, d*k
        self.Ba = {}
        # BaT : collection of matrix to compute hybrid part, d*k
        self.BaT = {}
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = {}

        self.updateCycle = 10
        self.lastUp = 0


        # other dicts to speed up computation
        self.AaIba = {}
        self.AaIBa = {}
        self.BaTAaI = {}
        self.theta = {}

        #additional stuff that is computated in the update function instead of recommend
	self.A0IBaTAaI = {}
        self.A0IBaA0IBaTAaI = {}

        self.lastart = []

        self.beta = np.zeros((self.k, 1))

        self.index = {}

        self.a_max = 0

        self.z = None
        self.zT = None
        self.xaT = None
        self.xa = None

    # Evaluator will call this function and pass the article features.
    # Check evaluator.py description for details.
    def set_articles(self, art):
        # init collection of matrix/vector Aa, Ba, ba
        i = 0
        art_len = len(art)
        self.art = art
        self.article_features = np.zeros((art_len, 1, self.d))
        self.Aa = np.zeros((art_len, self.d, self.d))
        self.AaI = np.zeros((art_len, self.d, self.d))
        self.ba = np.zeros((art_len, self.d, 1))
        self.AaIba = np.zeros((art_len, self.d, 1))
        self.AaIBa = np.zeros((art_len, self.d, self.k))
        self.theta = np.zeros((art_len, self.d, 1))
        for key in art:
            self.index[key] = i
            self.article_features[i] = art[key]
            self.Aa[i] = np.identity(self.d)
            self.AaI[i] = np.identity(self.d)
            self.ba[i] = np.zeros((self.d, 1))
            self.AaIba[i] = np.zeros((self.d, 1))
            self.theta[i] = np.zeros((self.d, 1))
            i += 1


    # This function will be called by the evaluator.
    # Check task description for details.
    def update(self, reward):
        global ttime
        #print reward
        if reward == -1:
             pass
        else:
            r = self.r[reward]
            self.Aa[self.a_max] += np.dot(self.xa, self.xaT)
            B = np.divide(
                np.dot(np.dot(self.AaI[self.a_max], np.dot(self.xa, self.xaT)), self.AaI[self.a_max]),
                (1 + np.dot(np.dot(self.xaT, self.AaI[self.a_max]), self.xa)))

            self.AaI[self.a_max] = self.AaI[self.a_max] - B
            self.ba[self.a_max] += r * self.xa
            self.AaIba[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
            
            self.theta = self.AaIba - np.dot(self.AaIBa, self.beta)
            self.lastart = []


    # This function will be called by the evaluator.
    # Check task description for details.
	# Use vectorized code to increase speed
    def recommend(self, timestamp, user_features, articles):
        global ttime
        article_len = len(articles)
        # za : feature of current user/article combination, k*1
        self.xaT = np.array([user_features])
        self.xa = np.transpose(self.xaT)
        # recommend using hybrid ucb
        # fast vectorized for loops

        index = [self.index[article] for article in articles]

        ttime = ttime - time.time()
        if cmp(self.lastart,index)!=0:
            self.article_features_tmp = self.article_features[index]

            #preprocess matrices
            self.preAaI = self.AaI[index]
            self.lastart = index
            self.preTheta = self.theta[index]
            
        ttime = ttime + time.time()
        AaIxa = np.dot(self.preAaI, self.xa)

	sa_tmp = np.transpose(np.dot(np.transpose(AaIxa,(0,2,1)),self.xa),(0,2,1))

        xaTtheta_tmp = np.transpose(np.dot(np.transpose(self.preTheta,(0,2,1)),self.xa),(0,2,1))

        max_index = np.argmax(xaTtheta_tmp + self.alpha * np.sqrt(sa_tmp))

        art_max = index[max_index]

        # article index with largest UCB
        # global a_max, entries
        self.a_max = art_max
       # return np.random.choice(articles)
        return articles[max_index]



algorithms = {
    'disjointUCB': disjointUCB()
}

algorithm = algorithms[method]
set_articles = algorithm.set_articles
update = algorithm.update
ttime = 0
recommend = algorithm.recommend
