"""
Search Results Relevance @ Kaggle
__author__ : Vijay Sathish

"""
import pandas as pd
import numpy as np
import math
import re
from sklearn import metrics, grid_search
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import (KFold, StratifiedKFold)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import (LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.stem.porter import *
from nltk.stem.snowball import *
from bs4 import BeautifulSoup

# Global parameters
gridSearch = True
cvFolds = 5
nJobs = 5
randomState = 30				# random Seed for various algorithms
cleanup = 'modified'		# Choice between original or modified cleanup

# Parameters for Model 1
stemmerEnableM1 = False
postStemmerM1 = True
predM1 = []

# Parameters for Model 2
stemmerType = 'snowball' 			# Choice of Porter or Snowball
predM2 = []

# Parameters for remaining models
stemmerEnableM3 = False
stemmerEnableM4 = False
stemmerEnableM5 = False
stemmerEnableM6 = False
stemmerEnableM7 = False
stemmerEnableM8 = False


### The following 3 functions to calculate Kappa score have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

### Evaluation metric for the contest
def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def perform_grid_search (pipeline, params, X, y) :
    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
		
    # Set up Stratified K-Fold CV
    skf = StratifiedKFold(labels, n_folds = cvFolds, random_state = randomState)
    
    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = pipeline, param_grid = params, scoring = kappa_scorer,
                                     verbose = 10, n_jobs = nJobs, iid = True, refit = True, cv = skf)
    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print ("Best estimator: ", model.best_estimator_)

    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return model

### Original cleanup but with wrong order of operations
def original_cleanup (train, test, stemmer) :
    train_data = []
    train_labels = []
    test_data = []
    
    print ("Performing Original Cleanup...")
    for i in range(len(train['query'])):
				# All query features are tagged with q, all title features are tagged with z
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
				# Replace anything apart from numbers and letters by space
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        train_data.append(s)
        # print ("S_data: ", s)
        train_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test['query'])):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        test_data.append(s)
    return train_data, train_labels, test_data

### Porter stemming seems to work fine even with pre-tagging, but stop words definitely wont with TFIDF
# Modified cleanup uses BS4, re.sub, stem with stop-words, and pre-tag join in that specific order
def modified_cleanup (data, stemmer, is_train = True, pretag = 'full') :
    s_data = []
    s_labels = []
    print ("Performing Modified Cleanup...")
    for i in range(len(data['query'])):
        query = (" ").join([z for z in BeautifulSoup(data["query"][i]).get_text(" ").split(" ")])
        title = (" ").join([z for z in BeautifulSoup(data["product_title"][i]).get_text(" ").split(" ")])
        description = (" ").join([z for z in BeautifulSoup(data["product_description"][i]).get_text(" ").split(" ")])
				
				## Replace anything apart from numbers and letters by space
				# TODO - 06/17 - Maybe we want to retain . and hyphen as well in the substitution
        query = re.sub("[^a-zA-Z0-9]"," ", query)
        title = re.sub("[^a-zA-Z0-9]"," ", title)
        description = re.sub("[^a-zA-Z0-9]"," ", description)

        ## Stemmer and join
        query = query.lower()
        title = title.lower()
        description = description.lower()
        query = (" ").join([stemmer.stem(z) for z in query.split(" ")])
        title = (" ").join([stemmer.stem(z) for z in title.split(" ")])
        description = (" ").join([stemmer.stem(z) for z in description.split(" ")])

				## Combine everything with pre-tagging
        if (pretag == 'ab_style') :
            bag_of_words = (" ").join([z for z in query.split(" ")]) + " " + (" ").join([z for z in title.split(" ")])
        elif (pretag == 'none') :					# Adding description w/o pre-tags is a BAD IDEA
            bag_of_words = (" ").join([z for z in query.split(" ")]) + " " + (" ").join([z for z in title.split(" ")]) + " " +  (" ").join([z for z in description.split(" ")])  
        elif (pretag == 'full') :					# original version
            bag_of_words = (" ").join(["q"+ z for z in query.split(" ")]) + " " + (" ").join(["z"+ z for z in title.split(" ")]) + " " +  (" ").join([z for z in description.split(" ")])  
        else : 	# partial pre-tag only tags description but in practice did worst of the lot
            bag_of_words = (" ").join([z for z in query.split(" ")]) + " " + (" ").join([z for z in title.split(" ")]) + " " +  (" ").join(["z" + z for z in description.split(" ")])  

        s_data.append(bag_of_words)
    return s_data

## Basic Lambda magic plus stemming based on options
def stemmer_clean (train, test, stemmer_enable = False, stemmer_type = 'porter'):
    trainData = []
    testData = []
    ## Lambda magic on text columns for concatenation
    train_fused = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    test_fused = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    ## Use Stemmer for this one too if enabled
    if (stemmer_enable) :
        if (stemmer_type == 'porter') :
            print ("NOTE: Porter Stemmer enabled for model M4...")
            stemmer = PorterStemmer()
        else :
            print ("NOTE: Snowball Stemmer enabled for model M4...")
            stemmer = SnowballStemmer('english')
						
        for i in range(len(train_fused)):
            s= (" ").join([stemmer.stem(z) for z in train_fused[i].split(" ")])
            trainData.append(s)
        for i in range(len(test_fused)):
            s= (" ").join([stemmer.stem(z) for z in test_fused[i].split(" ")])
            testData.append(s)
    else :					       # Copy as is
        trainData = train_fused
        testData = test_fused
    return trainData, testData
 

### Basic TruncatedSVD + SVM with RBF kernel model
def train_and_predict_m1 (train, test, labels) :
    print ("Training M1 (randomState = %d ...", randomState)
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM1, stemmer_type = 'porter')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    vectorizer = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 3), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    vectorizer.fit(trainData)
    X =  vectorizer.transform(trainData) 
    X_test = vectorizer.transform(testData)
    
    ## Use Stemmer post TF-IDF to check if things change
    # print (X)
    print ("X.shape: ", X.shape)
    print ("X_test.shape: ", X_test.shape)

    ## Create the pipeline 
		# 07/02 - RandomizedPCA/PCA does not work on sparse input (so cannot be applied on output of Vectorizer)
		# JimingYe says LDA did not give much benefit.
    clf = Pipeline([('svd', TruncatedSVD(random_state = randomState, n_components = 330)),
    						 						('scl', StandardScaler()),
                    	     ('svm', SVC(random_state = randomState, cache_size = 500, C = 12))])

    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
     param_grid = {'svd__n_components' : [200, 250, 300], 'svm__C': [10, 12]}
    # param_grid = {'svd__n_components' : [280], 'svm__C': [10]}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### Basic TruncatedSVD + SVM model with modified cleanup
def train_and_predict_m2 (train, test, labels) :
    print ("Training M2...")
    trainData = []
    testData = []
    trainLabels = []
    # Remove html, remove non text or numeric, make query and title unique features for counts using prefix
    if (stemmerType == 'porter') :
        stemmer = PorterStemmer()
    else :
        stemmer = SnowballStemmer("english")

		# Beautiful soup cleanup and stemming
    if (cleanup == 'original') :
		    trainData, trainLabels, testData = original_cleanup(train, test, stemmer)
    else :
		    trainData = modified_cleanup(train, stemmer, is_train = True)
		    testData = modified_cleanup(test, stemmer, is_train = False)
				
    vectorizer = TfidfVectorizer(min_df = 5, max_df = 500, max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 3), sublinear_tf = True, stop_words = ML_STOP_WORDS)
    vectorizer.fit(trainData)
    X =  vectorizer.transform(trainData) 
    X_test = vectorizer.transform(testData)

		# SVD defaults: algorithm='randomized', n_iter=5 
    clf = Pipeline([('svd', TruncatedSVD(random_state = randomState, n_components = 260)), 
                   ('scl', StandardScaler()), 
                   ('svm', SVC(random_state = randomState, cache_size = 500, C = 10))])
    
		## Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [400, 300, 200], 'svm__C': [10]}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### M3 attempts SGD without any SVD or StandardScaling
def train_and_predict_m3 (train, test, labels) :
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM3, stemmer_type = 'porter')

    """
    # Beautiful soup cleanup and stemming
    stemmer = PorterStemmer()
    trainData = modified_cleanup(train, stemmer, is_train = True)
    testData = modified_cleanup(test, stemmer, is_train = False)
    """
				
    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 6), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    clf = SGDClassifier(random_state = randomState, n_jobs = 1, penalty = 'l2', loss = 'huber', n_iter = 50, class_weight = 'auto', learning_rate = 'optimal', epsilon = 1)
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'n_iter' : [30, 50, 80, 100, 200],  'loss': ['huber'], 'epsilon' : [0.3, 1], 'alpha' : [0.0001, 0.0003, 0.001] }
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred


### M4 attempts LogisticRegression without SVD or Standard Scaling
def train_and_predict_m4 (train, test, labels) :
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM4, stemmer_type = 'porter')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 6), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    clf = LogisticRegression(random_state = randomState, penalty = 'l2', C = 12, class_weight = 'auto')
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
    #param_grid = {'C' : [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 30], 'penalty' : ['l2']}
    param_grid = {'C' : [1, 3, 5, 6, 7, 8, 10, 11, 12], 'penalty' : ['l2']}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### M5 attempts Multinomial Naive Bayes
def train_and_predict_m5 (train, test, labels) :
		# Beautiful soup cleanup and stemming (just to mix it up)
    stemmer = PorterStemmer()
    trainData = modified_cleanup(train, stemmer, is_train = True, pretag = 'full')
    testData = modified_cleanup(test, stemmer, is_train = False, pretag = 'full')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 3), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    print ("Fitting Multinominal Naive Bayes...")
    clf = MultinomialNB(alpha = 0.03)
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
    # param_grid = {'alpha' : [0.01, 0.03, 0.1, 0.3, 1]}
    param_grid = {'alpha' : [0.01, 0.03]}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### M6 attempts K-Nearest Neighbors
def train_and_predict_m6 (train, test, labels) :
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM6, stemmer_type = 'snowball')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 3,  max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 3), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    print ("Fitting K-Nearest Neighbors...")
    clf = KNeighborsClassifier(p = 2, n_neighbors = 5)
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
		# Note: minkowski with p > 2 does not work for sparse matrices
    param_grid = {'n_neighbors' : [3, 4, 5, 6, 7], 'weights' : ['uniform', 'distance'], 'leaf_size' : [1, 3, 5, 10] }
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### M7 attempts Passive Aggressive Classifier
# Looks like the 3rd strongest classifier!
def train_and_predict_m7 (train, test, labels) :
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM7, stemmer_type = 'snowball')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 5, max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 5), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    print ("Fitting Passive-Aggressive Classifer...")
    clf = PassiveAggressiveClassifier(random_state = randomState, loss = 'squared_hinge', n_iter = 100, C = 0.01)
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
		# Note: minkowski with p > 2 does not work for sparse matrices
    param_grid = {'C' : [0.003, 0.01, 0.03, 0.1], 'loss': ['hinge', 'squared_hinge'], 'n_iter': [5, 10, 30, 100, 300]}
    #param_grid = {'C' : [0.003, 0.01, 0.03, 0.1, 0.3, 1], 'loss': ['hinge'], 'n_iter': [5, 10, 30, 100, 300, 1000]}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

### M8 attempts a RidgeClassifier
def train_and_predict_m8 (train, test, labels) :
    ## Apply basic concatenation + stemming
    trainData, testData = stemmer_clean (train, test, stemmerEnableM7, stemmer_type = 'porter')

    ## TF-IDF transform with sub-linear TF and stop-word removal
    tfv = TfidfVectorizer(min_df = 5, max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 5), smooth_idf = 1, sublinear_tf = 1, stop_words = ML_STOP_WORDS)
    tfv.fit(trainData)
    X =  tfv.transform(trainData) 
    X_test = tfv.transform(testData)
    
    ## Create the classifier
    print ("Fitting Ridge Classifer...")
    clf = RidgeClassifier(class_weight = 'auto', alpha = 1, normalize = True)
    
    ## Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'alpha' : [0.1, 0.3, 1, 3, 10], 'normalize' : [True, False]}
    
    ## Predict model with best parameters optimized for quadratic_weighted_kappa
    if (gridSearch) :
        model = perform_grid_search (clf, param_grid, X, labels)    	
        pred = model.predict(X_test)
    else :
        clf.fit(X, labels)    	
        pred = clf.predict(X_test)
    return pred

if __name__ == '__main__':
    ## Stopwords Tweak
    sw = []
    stemmer = PorterStemmer()
    ML_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height']
    ML_STOP_WORDS += list(text.ENGLISH_STOP_WORDS)
    for stw in ML_STOP_WORDS:
        sw.append('z'+str(stw))
        sw.append('q'+str(stw))
    ML_STOP_WORDS += sw

    for i in range(len(ML_STOP_WORDS)):
        ML_STOP_WORDS[i] = stemmer.stem(ML_STOP_WORDS[i])

    ## Pre-processing input
    train = pd.read_csv("D:/Kaggle/CFlowerSR/input/train.csv").fillna("")
    test = pd.read_csv("D:/Kaggle/CFlowerSR/input/test.csv").fillna("")
    
    ## Store ids for submission file
    idx = test.id.values.astype(int)
    
    ## Create labels
    labels = train.median_relevance.values
    
		## Drop unwanted columns
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

		## Invoke the various classifiers
    randomState = 30
    predM1 = np.array(train_and_predict_m1 (train, test, labels))
    predM2 = np.array(train_and_predict_m2 (train, test, labels))
    raw = pd.DataFrame({"id": idx, "prediction": predM1})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R1.csv", index=False)
    raw = pd.DataFrame({"id": idx, "prediction": predM2})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R2.csv", index=False)
    
    randomState = 6890
    predM3 = np.array(train_and_predict_m1 (train, test, labels))
    predM4 = np.array(train_and_predict_m2 (train, test, labels))
    raw = pd.DataFrame({"id": idx, "prediction": predM3})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R3.csv", index=False)
    raw = pd.DataFrame({"id": idx, "prediction": predM4})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R4.csv", index=False)
    
    randomState = 0
    predM5 = np.array(train_and_predict_m1 (train, test, labels))
    predM6 = np.array(train_and_predict_m2 (train, test, labels))
    raw = pd.DataFrame({"id": idx, "prediction": predM5})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R5.csv", index=False)
    raw = pd.DataFrame({"id": idx, "prediction": predM6})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R6.csv", index=False)

    randomState = 10911
    predM7 = np.array(train_and_predict_m1 (train, test, labels))
    predM8 = np.array(train_and_predict_m2 (train, test, labels))
    raw = pd.DataFrame({"id": idx, "prediction": predM7})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R7.csv", index=False)
    raw = pd.DataFrame({"id": idx, "prediction": predM8})
    raw.to_csv("D:/Kaggle/CFlowerSR/random_R8.csv", index=False)
		
    assert (predM1.shape == predM2.shape)
    preds = np.column_stack([predM1, predM2, predM3, predM4, predM5, predM6, predM7, predM8])
    print ("preds.shape: ", preds.shape)
    test_pred_raw = pd.DataFrame(preds, columns = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'])
    test_pred_raw.to_csv("D:/Kaggle/CFlowerSR/random8_modelsm1m2_ngram13_test_pred_raw_cv5.csv", index=False)
	
		"""
    pred_stacked_round = (np.rint(np.mean(preds, axis = 1))).astype(np.int)
    print ("pred_stacked_round.shape", pred_stacked_round.shape)
    ## Create submission file
    submission = pd.DataFrame({"id": idx, "prediction": pred_stacked_round})
    submission.to_csv("D:/Kaggle/CFlowerSR/results/random4_models8_m1m2_ngram13_mean_round_pred_cv5.csv", index=False)
		"""
