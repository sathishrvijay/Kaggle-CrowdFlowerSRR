"""
Blending model 
Search Results Relevance @ Kaggle
__author__ : Vijay Sathish

"""
import pandas as pd
import numpy as np
import math
import re
from sklearn.linear_model import (LogisticRegression, SGDClassifier)
from sklearn import metrics, grid_search
from sklearn.cross_validation import (KFold, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import (TruncatedSVD, NMF, RandomizedPCA)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.stem.porter import *
from nltk.stem.snowball import *
from bs4 import BeautifulSoup
# array declaration

numModels = 16			# value of either 4 or 8 right now
roundAvgPreds = True
exaggerateWeights = 3			# How much to scale weights by
votingClassifer = False		# Whether to use averaging or voting classifier to make final prediction


## Take a set of predictions and return final prediction by majority voting
def get_majority_rating (votes) :
	# The best models get preference most of the time unless the weak ones can collectively crowd it out
	#weights = [0.609, 0.593, 0.551, 0.560, 0.529, 0.561, 0.576, 0.488]
	weights =  [12,    9,      2, 		2, 		 1, 		2, 		  3, 			1 ]
	expand_votes = []
	vote_counts = np.zeros(4)			#1, 2, 3, 4 are the only possible ratings
	for i, v in enumerate(weights) :
		expand_votes.extend([votes[i] for x in range(weights[i])])
	#print ("Expanded votes: ", expand_votes)
	## Count #votes per rating
	for i in range(vote_counts.shape[0]) :
		vote_counts[i] = len([1 for rating in expand_votes if (rating == i+1)])
	#print ("Final vote counts: ", vote_counts)

	## Pick rating by majority vote
	## In case of ties, favor the higher rating because we know higher rating is more likely
	majority_rating = 0
	final_rating = 1
	for idx, count in enumerate(vote_counts) :
		if (majority_rating <= count) :
			majority_rating = count
			final_rating = idx + 1			# Ratings are 1, 2, 3 and 4; hence adjust idx
	#print ("Final rating: ", final_rating)
	return final_rating

if __name__ == '__main__':
	pred_stacked_floor = []
	pred_stacked_round = []
	pred_voting = []
	final_preds = []

	## Pre-processing input
	test = pd.read_csv("D:/Kaggle/CFlowerSR/input/test.csv").fillna("")
	if (numModels == 4) :
		weights = [0.616, 0., 0.54, 0.50]
	elif (numModels == 8) :
		#preds = pd.read_csv("D:/Kaggle/CFlowerSR/models8_tuned_test_pred_raw_cv5.csv")
		#preds = pd.read_csv("D:/Kaggle/CFlowerSR/random8_modelsm1m2_ngram12_test_pred_raw_cv5.csv")
		preds = pd.read_csv("D:/Kaggle/CFlowerSR/random8_modelsm1m2_ngram13_test_pred_raw_cv5.csv")
		# weights = [0.616, 0.595, 0.614, 0.588, 0.618, 0.587, 0.621, 0.591]
		weights = [0.604, 0.588, 0.601, 0.589, 0.611, 0.587, 0.605, 0.584]
		#weights = [5,      4,    2,     2,      0.5,     2,     3,     0.5]
	else :
		preds = pd.read_csv("D:/Kaggle/CFlowerSR/random16_modelsm1m2_ngram123_test_pred_raw_cv5.csv")
		weights = [0.616, 0.595, 0.614, 0.588, 0.618, 0.587, 0.621, 0.591, 0.604, 0.588, 0.601, 0.589, 0.611, 0.587, 0.605, 0.584]

	## Store ids for submission file
	idx = test.id.values.astype(int)

	## Exaggeration of weights
	if (votingClassifer) : 
		for row in range(preds.shape[0]) :
			pred_voting.append(get_majority_rating(preds.iloc[row]))
		pred_voting = np.array(pred_voting)
		print ("pred_voting.shape", pred_voting.shape)
		final_preds = pred_voting
	else :
		print ("Calculating weighted(^%d) averaging output..." %(exaggerateWeights))
		for i, w in enumerate(weights) :
			weights[i] = w ** exaggerateWeights		# power
		print ("Weights: ", weights)

		# Weighted average
		pred_stacked_floor = (np.floor(np.average(preds, axis = 1, weights = weights))).astype(np.int)
		print ("pred_stacked_floor.shape", pred_stacked_floor.shape)
		pred_stacked_round = (np.rint(np.average(preds, axis = 1, weights = weights))).astype(np.int)
		print ("pred_stacked_round.shape", pred_stacked_round.shape)
		if (roundAvgPreds) :
			print ("Writing rounded output...")
			final_preds = pred_stacked_round
		else :
			print ("Writing floored output...")
			final_preds = pred_stacked_floor
	
	## Write final predictions to CSV
	submission = pd.DataFrame({"id": idx, "prediction": final_preds})
	submission.to_csv("D:/Kaggle/CFlowerSR/results/random4_models16_m1m2_ngram123_meanexa3_round_pred_cv5.csv", index=False)



