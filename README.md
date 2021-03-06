# Kaggle - Crowd Flower Search Results Relevance competition

### Final Standing
- 428/1326 (top 3rd) in private leaderboard

### Problem Statement
- The goal of this competition was to look at search query for a product and the result shown and rank the result on a scale of 1-4 based on relevance to the query. The train and test data included the query phrase, product title and uncleaned description. The training data had an additional search relevance rating column. 
- Hence, it boils down to a text mining challenge with supervised classification

### Files 
- classifier.py - Truncated SVD + SVM classifier prediction models
- blend_post_process.py - Post processing scripts to combine predictions from different classifiers

### Models and Tuning
- Borrowed weighted Kappa scorer code from competition admin Ben Hammer's git repo
- Basic classifier model inspired from Kaggle Master Abhishek
- Initial pre-processing step involved cleaning up the reviews:
-   This involved using Beautiful Soup to remove all the html tags, some regex to remove special characters, conversion to lower case, removal of stop words and Porter/Snowball stemmers to calculate stem words
- Second step involved Term Frequency - Inverse Document Frequency weighting (aka TFIDF) transform of stemmed words to create features
-   1-2 and 1-3 word ngrams were used as features. Seemed that larger ngrams were reducing accuracy. 
-   Most likely cause is because most queries were only 3-4 words long, so larger ngrams were causing overfitting on training data
- Third step was to convert sparse output of TfidfVectorizer to dense output by using TruncatedSVD. 
-   Truncated SVD is similar to PCA but works on sparse input
- Fourth step was to feed the SVD output to a classifier to train and predict search result relevance rating
-   Found that Support Vector Classifier with RBF kernel worked the best

### Feature Engineering
- Most important feature engineering steps were the review cleanup, stop word removal, stemming, TFIDF + SVD transform
- Surprising, concatenating query and product title and description worked well by itself without any fancy manipulations

### Final Model 
- Was a weighted average of 16 models that were variations of two base models
-   First base model predicted based on only the search query and product title
-   Second base model predicted based on the search query, product title and product description
-   16 models were derived from these base models by changing random seeds, ngram range etc...
-   Models were weighted by cube of public leaderboard submission score to give higher preference to better performing models and rounding the average result


### What didn't work
- Logistic Regression, KNearest Neighbours, Stochastic Gradient Descent, Multinomial NB etc gave decent scores but nothing close to the SVD + SVC combo, so didn't use in the final ensemble
- Randomized PCA, Non Negative Matrix Factorization did not work instead of SVD since they don't work on sparse input features
- Restricted Boltzmann Machines was attempted for Dimensionality Reduction, but it looked like it was way too slow and performance was really poor compared to SVD

### Could have tried / Winning solutions
- Using Ordinal classifiers that take ranking into consideration might have improved performance over vanilla classifiers
- According to winners' solution, the correlation or distance between the search query and product title/description turned out to be among the most useful features. This included metrics like Jaccard coefficients, dice distance, intersect word counts etc.
- Data cleaning to correct spelling mistakes and replace with synonyms etc. (Non-trivial effort to do this)
- Turns out that using Gradient Boosting Regressor with linear booster as opposed to a Classifier was actually one of the strongest predictor model with a simple objective function like MSE as opposed to weighted Kappa scoring specified in the competition.
- Another key insight was to use the Cumulative Distributive Function to find percentage of results ranked 1, 2, 3 and 4 in training set. Then, in the predictions, the winner first generated ranking predictions and then used CDF percentages of the training set to convert ranking predictions to ratings from 1 to 4 which worked really well!
 



