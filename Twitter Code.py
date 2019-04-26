#LOADING LIBRARIES

import re    # for regular expressions 
import nltk  # for text manipulation 
import string # forString operations (not necessary)
from datetime import datetime # To access datetime
import warnings # for throwing exceptions
import numpy as np # for scientific computing
import pandas as pd # for working with data
import seaborn as sns # extension of matplotlib
import matplotlib.pyplot as plt #plots graphs
from sklearn.preprocessing import StandardScaler #To transform the data

from nltk.stem.porter import * #To use PorterStemmer function
from wordcloud import WordCloud #To use wordcloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #TfidfVectorizer->gives higher weight to words occuring rare in a entire corpus but good in few documents
import gensim #Topic modelling(identify which topic is discussed), Similarity retrieval

from tqdm import tqdm
tqdm.pandas(desc="progress-bar") #To display progress bar with title "progres-bar" 
from gensim.models.deprecated.doc2vec import LabeledSentence #Labelling tweets for doc2vec purpose

from sklearn.linear_model import LogisticRegression # To build Logistic regression model
from sklearn.model_selection import train_test_split # For splitting data into train and test data
from sklearn.metrics import f1_score # To compute performance of the model
from sklearn import svm # To build Support vector machine model
from sklearn.ensemble import RandomForestClassifier #To build Random forest classifier model
from xgboost import XGBClassifier #To build extreme gradient boosting model
import xgboost as xgb #Imports all features of extreme gradient boosting algorithm
import lightgbm as lgb #To build light gradient boosting model

np.random.seed(11)#To reproduce results

###################################################################################################################################################################

#DATA INSPECTION

#sets value
pd.set_option("display.max_colwidth", 200) 
#To ignore deprecation  warnings
warnings.filterwarnings("ignore")


#importing datasets
train  = pd.read_csv('train_tweets.csv') 
test = pd.read_csv('test_tweets.csv')


#first 10 data of non-racist and racist tweets resp..,
print("\nNON RACIST TWEETS\n")
print(train[train['label'] == 0].head(10))
print("\nRACIST TWEETS\n")
print(train[train['label'] == 1].head(10))


#dimensions of data set
print("\nTRAINING SET\n")
print(train.shape)
print("\nTEST SET\n")
print(test.shape)


#split of tweets in terms of labels in training set
print("NO OF POSITIVE AND NEGATIVE TWEETS IN TRAINING DATASET")
print(train["label"].value_counts())


#tweets in terms of number of words in each tweet
length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len()
    #histogram autoamatically scales dataset suitable to plot the graph
    #here bins seperate the enitre dataset into intervals of 20 and plot graph (discrete kind)
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.legend()
plt.xlabel("Tweet id")
plt.ylabel("No of Words")
plt.show()


###################################################################################################################################################################

#DATA CLEANING

#To combine train and test data for cleaning
combi = train.append(test, ignore_index=True , sort=False) 


#user-defined function to remove unwanted text patterns from the tweets.
def remove_pattern(input_txt, pattern):
    #Finds all words matching the pattern
    r = re.findall(pattern, input_txt)
    for i in r:
        #removes the matched words
        input_txt = re.sub(i, '', input_txt)
    return input_txt


#Removing user handles (\w - words)
    #vectorize function used when recursively a function is called
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 


#Removing Punctuations, Numbers, and Special Characters (^ - except)
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 


#Removing short words (assuming that less than 3 letter words will not much influence over sentiment)
    #lambda function is similar to macros
    #apply funcion applies particular function over every element
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


#Tokenization (List for each tweet where items are each word in the tweet)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


#Normalization 
tokenized_tweet = tokenized_tweet.apply(lambda x: [PorterStemmer().stem(i) for i in x])
print("\nFIRST 5 PROCESSED TOKENIZED TWEETS\n")
print(tokenized_tweet.head())

#Stitching normalized tokens together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet


###################################################################################################################################################################


#VISUALIZATION FROM TWEETS

#word cloud visualization is used to identify frequency of words

#Non-racist tweets
    #Taking non - racist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
    #Generating word cloud
wordcloud = WordCloud(width=800, height=500, random_state=11, max_font_size=110).generate(normal_words)
    #Plotting word cloud in graph
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="none")
plt.axis('off')
plt.show()


#Racist tweets
    #Taking non - racist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
    #Generating word cloud
wordcloud = WordCloud(width=800, height=500, random_state=11, max_font_size=110).generate(normal_words)
    #Plotting word cloud in graph
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="none")
plt.axis('off')
plt.show()


#Function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        #r - raw string used for specifying regular expressions
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# extracting hashtags from non-racist/sexist tweets 
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])


# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])


# unnesting list (to make muliple list as single list)
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


#Plotting non-racist hashtags
    #Frequency of each item
a = nltk.FreqDist(HT_regular)
    #Storing frequency as dict/2D form
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
    #Plotting
plt.figure(figsize=(18,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
plt.show()


#Plotting racist hashtags
    #Frequency of each item
a = nltk.FreqDist(HT_negative)
    #Storing frequency as dict/2D form
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
    #Plotting
plt.figure(figsize=(18,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
plt.show()



###################################################################################################################################################################

#Bag-of-words Features
    #max_df/min_df = int->no of documents ; float->percentage among total documents
    #max_features = top 1000 features taken
    #stop_words = english-> inbulit stop words list for english is used
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')


#making data to center with mean zero and unit standard deviation
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])



###################################################################################################################################################################

#TF-IDF Features
    #TF = (Number of times term t appears in a document)/(Number of terms in the document)
    #IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
    #TF-IDF = TF*IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')


#making data to center withe mean zero and unit standard deviation
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


###################################################################################################################################################################


#_____________________________________________________________Notes on Word2Vec________________________________________________________________________________
    #Word embeddings-> representing words as vectors
    #               -> high dimensional word features into low dimensional feature vectors by preserving the contextual similarity

    
    #Combination of  CBOW (Continuous bag of words) and Skip-gram model.
        #CBOW-> tends to predict the probability of a word given a context
        #Skip-gram model-> tries to predict the context for a given word.


    #Softmax-> converts vector as probability distribution

    
    #Pretrained Word2Vec models (huge in size)
        #Google News Word Vectors
        #Freebase names
        #DBPedia vectors (wiki2vec)
    




#Training own Word2Vec model
    #tokenizing
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # to represent in no.of.dimensions-> more the dimension, more the efficiency of model
            window=5, # takes 10 words surrounding the current word to find context of current word 
            min_count=2, # removes words with frequency less than 2
            sg = 1, # 1 for skip-gram model
            hs = 0, # 0 for negative sampling
            negative = 10, # only takes random 10 negative samples(since dataset is huge, this is done)
            workers= 1, # no.of threads to train the model - set to 1 to reproduce same results
            seed = 11, # to generate same random numbers every time
            hashfxn = hash # for reproducability
            )  


#Training the built model-> should specify model, size of corpus, epoch
model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)


#Getting similar context words to the mentioned word
sim_dinner = model_w2v.wv.most_similar(positive="dinner")
sim_trump = model_w2v.wv.most_similar(positive="trump")


i=0
sim_dinner_len = len(sim_dinner)
print("\nWORDS SIMILAR TO DINNER\n")
while(i<sim_dinner_len):
    print("\n")
    print(sim_dinner[i])
    i=i+1
i=0
print("\nWORDS SIMILAR TO TRUMP\n")
sim_trump_len = len(sim_trump)
while(i<sim_trump_len):
    print("\n")
    print(sim_trump[i])
    i=i+1





#Functions for converting tweets into vectors
def word_vector(tokens, size):
    #Creates array with specified all filled with zeros and it is given a new shape
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            #every token in a tweet is converted as a vector
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        # handling the case where the token is not in vocabulary continue
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


#word2vec feature set
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
#converting array into 2D Table
wordvec_df = pd.DataFrame(wordvec_arrays)



###################################################################################################################################################################

#Doc2vec Embedding

#To label each tweet
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        #Tweet Id's are itself made as label
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet)
#Prints o to 5 labeled tweets
print("\nFIRST 6 LABELED TWEETS FOR ILLUSTRATION\n")
print(labeled_tweets[:6])


#Doc2vec is a small extension to the CBOW model.
#Instead of using just words to predict the next word, we also added paragraph ID which is document-unique.


#Training doc2vec model
model_d2v = gensim.models.Doc2Vec(
            dm=1, # dm = 1 for ‘distributed memory’ model
            dm_mean=1, # dm = 1 for using mean of the context word vectors
            vector_size=200, # to represent in no.of.dimensions-> more the dimension, more the efficiency of model                                 
            window=5, # takes 10 words surrounding the current word to find context of current word                                   
            negative = 7, # only takes random 7 negative samples(since dataset is huge, this is done)
            min_count=2, # removes words with frequency less than 2                                  
            workers=1, # no. of threads to train the model - to reproduce same results                                  
            alpha=0.1, # learning rate                                  
            seed = 11, # to generate same random numbers every time
            ) 


#To show progress bar while training Doc2vec model
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
#Training Doc2vec model
model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)


#Preparing doc2vec Feature Set
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    
#converting array into 2D Table
docvec_df = pd.DataFrame(docvec_arrays) 


###################################################################################################################################################################


#Building different machine learning models to use later
    #We are computing evaluation time for different models 

# Creating Logistic regression model 
    #solver -> algorithm for optimimzation problem
lreg = LogisticRegression(solver = "lbfgs",  random_state=11)

# Creating SVM model
    #kernel -> type of classifier
    #C -> regularization ; lower value of c, larger is the margin seperating hyperplane 
svc = svm.SVC(kernel='linear', C=1, random_state = 11, probability=True)

#Creating Random Forest model
    #n_estimators -> No of trees
rf = RandomForestClassifier(n_estimators=400, random_state=11)

#Creating xgb model
    #max_depth -> tree depth ; more the depth, more is the complexity
xgb_cl = XGBClassifier(random_state=11, n_estimators=1000)


#We are going to split the bow, tfidf, word2vec and doc2vec features into train and test data once (in Logitic regression model)
#Use the same train and test data correspondingly in all other machine leaning models also

###################################################################################################################################################################

print("\n\nLOGISTIC REGRESSION MODEL ACCURACIES\n\n")

#BOW FEATURES - Logistic Regression model 
# Segregating dataset into train and test BoW features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting Training dataset into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=11,test_size=0.3) 
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
lreg.fit(xtrain_bow, ytrain)
stop = datetime.now()
lreg_bow_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING LOGISTIC REGRESSION\n")
print(bow_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
bow_lreg_test_pred = lreg.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_lreg_test_pred_int = bow_lreg_test_pred[:,1]>=0.3
bow_lreg_test_pred_int = bow_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_bow.csv', index=False)





#TF-IDF FEATURES - Logistic Regression model
# Segregating dataset into train and test TF-IDF features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
lreg.fit(xtrain_tfidf, ytrain)
stop = datetime.now()
lreg_tfidf_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING LOGISTIC REGRESSION\n")
print(tfidf_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
tfidf_lreg_test_pred = lreg.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_lreg_test_pred_int = tfidf_lreg_test_pred[:,1]>=0.3
tfidf_lreg_test_pred_int = tfidf_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_tfidf.csv', index=False)





#WORD2VEC FEATURES - Logistic Regression model
# Segregating dataset into train and test WORD2VEC features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
    #iloc - access data by row index(since wordvec_df is 2D)
train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
lreg.fit(xtrain_w2v, ytrain)
stop = datetime.now()
lreg_w2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING LOGISTIC REGRESSION\n")
print(w2v_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
w2v_lreg_test_pred = lreg.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_lreg_test_pred_int = w2v_lreg_test_pred[:,1]>=0.3
w2v_lreg_test_pred_int = w2v_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_w2v.csv', index=False)





#DOC2VEC FEATURES - Logistic Regression model
# Segregating dataset into train and test DOC2VEC features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
    #iloc - access data by row index(since docvec_df is 2D)
train_d2v = docvec_df.iloc[:31962,:]
test_d2v = docvec_df.iloc[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_d2v = train_d2v.iloc[ytrain.index,:]
xvalid_d2v = train_d2v.iloc[yvalid.index,:]
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
lreg.fit(xtrain_d2v, ytrain)
stop = datetime.now()
lreg_d2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING LOGISTIC REGRESSION\n")
print(d2v_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
d2v_lreg_test_pred = lreg.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_lreg_test_pred_int = d2v_lreg_test_pred[:,1]>=0.3
d2v_lreg_test_pred_int = d2v_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_d2v.csv', index=False)



###############################################################################################################################################################################

print("\n\nSUPPORT VECTOR MACHINE MODEL ACCURACIES\n\n")

#BOW FEATURES - SVM model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
svc.fit(xtrain_bow, ytrain)
stop = datetime.now()
svm_bow_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING SUPPORT VECTOR MACHINE\n")
print(bow_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
bow_svm_test_pred = svc.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_svm_test_pred_int = bow_svm_test_pred[:,1]>=0.3
bow_svm_test_pred_int = bow_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_bow.csv', index=False)





#TF-IDF FEATURES - SVM model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
svc.fit(xtrain_tfidf, ytrain)
stop = datetime.now()
svm_tfidf_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING SUPPORT VECTOR MACHINE\n")
print(tfidf_svm_pred_score)

#Testing the built SVM on test data

#predicting on test data
tfidf_svm_test_pred = svc.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_svm_test_pred_int = tfidf_svm_test_pred[:,1]>=0.3
tfidf_svm_test_pred_int = tfidf_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_tfidf.csv', index=False)





#WORD2VEC FEATURES - SVM model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
svc.fit(xtrain_w2v, ytrain)
stop = datetime.now()
svm_w2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING SUPPORT VECTOR MACHINE\n")
print(w2v_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
w2v_svm_test_pred = svc.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_svm_test_pred_int = w2v_svm_test_pred[:,1]>=0.3
w2v_svm_test_pred_int = w2v_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_w2v.csv', index=False)





#DOC2VEC FEATURES - SVM model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
svc.fit(xtrain_d2v, ytrain)
stop = datetime.now()
svm_d2v_exec_time = stop - start
# prediction on the validation set from Training dataset 
prediction = svc.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING SUPPORT VECTOR MACHINE\n")
print(d2v_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
d2v_svm_test_pred = svc.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_svm_test_pred_int = d2v_svm_test_pred[:,1]>=0.3
d2v_svm_test_pred_int = d2v_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_d2v.csv', index=False)


###########################################################################################################################################################################################

print("\n\nRANDOM FOREST MODEL ACCURACIES\n\n")

#BOW FEATURES - RANDOMFOREST model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
rf.fit(xtrain_bow, ytrain)
stop = datetime.now()
rf_bow_exec_time = stop - start

# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING RANDOM FOREST\n")
print(bow_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
bow_rf_test_pred = rf.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_rf_test_pred_int = bow_rf_test_pred[:,1]>=0.3
bow_rf_test_pred_int = bow_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_bow.csv', index=False)





#TF-IDF FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
rf.fit(xtrain_tfidf, ytrain)
stop = datetime.now()
rf_tfidf_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING RANDOM FOREST\n")
print(tfidf_rf_pred_score)

#Testing the built RANDOM FOREST on test data

#predicting on test data
tfidf_rf_test_pred = rf.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_rf_test_pred_int = tfidf_rf_test_pred[:,1]>=0.3
tfidf_rf_test_pred_int = tfidf_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_tfidf.csv', index=False)





#WORD2VEC FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
rf.fit(xtrain_w2v, ytrain)
stop = datetime.now()
rf_w2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING RANDOM FOREST\n")
print(w2v_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
w2v_rf_test_pred = rf.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_rf_test_pred_int = w2v_rf_test_pred[:,1]>=0.3
w2v_rf_test_pred_int = w2v_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_w2v.csv', index=False)





#DOC2VEC FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
rf.fit(xtrain_d2v, ytrain)
stop = datetime.now()
rf_d2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING RANDOM FOREST\n")
print(d2v_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
d2v_rf_test_pred = rf.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_rf_test_pred_int = d2v_rf_test_pred[:,1]>=0.3
d2v_rf_test_pred_int = d2v_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_d2v.csv', index=False)

#######################################################################################################################################################################################################


print("\n\nEXTREME GRADIENT BOOSTING MODEL ACCURACIES\n\n")

#BOW FEATURES - EXTREME GRADIENT BOOSTING model
 #Training the model with training set from Training dataset and computing training time
start = datetime.now()
xgb_cl.fit(xtrain_bow, ytrain)
stop = datetime.now()
xgb_bow_exec_time = stop - start
 #prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING EXTREME GRADIENT BOOSTING\n")
print(bow_xgb_pred_score)

#Testing the built EXTREME GRADIENT BOOSTING model on test data

#predicting on test data
bow_xgb_test_pred = xgb_cl.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_xgb_test_pred_int = (bow_xgb_test_pred[:,1])>=0.3
bow_xgb_test_pred_int = bow_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_bow.csv', index=False)





#TF-IDF FEATURES - EXTREME GRADIENT BOOSTING model
 #Training the model with training set from Training dataset and computing training time
start = datetime.now()
xgb_cl.fit(xtrain_tfidf, ytrain)
stop = datetime.now()
xgb_tfidf_exec_time = stop - start
 #prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING EXTREME GRADIENT BOOSTING\n")
print(tfidf_xgb_pred_score)

#Testing the built EXTREME GRADIENT BOOSTING on test data

#predicting on test data
tfidf_xgb_test_pred = xgb_cl.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_xgb_test_pred_int = (tfidf_xgb_test_pred[:,1])>=0.3
tfidf_xgb_test_pred_int = tfidf_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_tfidf.csv', index=False)





#WORD2VEC FEATURES - EXTREME GRADIENT BOOSTING model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
xgb_cl.fit(xtrain_w2v, ytrain)
stop = datetime.now()
xgb_w2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING EXTREME GRADIENT BOOSTING\n")
print(w2v_xgb_pred_score)

#Testing the built EXTREME GRADIENT BOOSTING model on test data

#predicting on test data
w2v_xgb_test_pred = xgb_cl.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_xgb_test_pred_int = (w2v_xgb_test_pred[:,1])>=0.3
w2v_xgb_test_pred_int = w2v_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_w2v.csv', index=False)





#DOC2VEC FEATURES - EXTREME GRADIENT BOOSTING model
# Training the model with training set from Training dataset and computing training time
start = datetime.now()
xgb_cl.fit(xtrain_d2v, ytrain)
stop = datetime.now()
xgb_d2v_exec_time = stop - start
# prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING EXTREME GRADIENT BOOSTING\n")
print(d2v_xgb_pred_score)

#Testing the built EXTREME GRADIENT BOOSTING model on test data

#predicting on test data
d2v_xgb_test_pred = xgb_cl.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_xgb_test_pred_int = (d2v_xgb_test_pred[:,1])>=0.3
d2v_xgb_test_pred_int = d2v_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_d2v.csv', index=False)


###################################################################################################################################################################

#Creating lgb model 
lgb_params ={
    #specifying laerning objective -> binary classification
    'objective':'binary',
    #Rate of training
    'learningRate':0.1,
    #tree depth ; more the depth, more is the complexity
    'max_depth':6,
    #type of boosting to be performed 
    'boosting_type':'gbdt',
    #No of leaves-> To control complexity
    'num_leaves':50,
    #Minimum data each leaf may have
    'min_data_in_leaf':20
    }


print("\n\nLIGHT GRADIENT BOOSTING MODEL ACCURACIES\n\n")

#BOW FEATURES - LIGHT GRADIENT BOOSTING model

#LGB accepts float32/64 input only
ytrain = ytrain.astype('float32')
xtrain_bow = xtrain_bow.astype('float64')
xvalid_bow = xvalid_bow.astype('float64')
xtrain_tfidf = xtrain_tfidf.astype('float64')
xvalid_tfidf = xvalid_tfidf.astype('float64')
xtrain_w2v = xtrain_w2v.astype('float64')
xvalid_w2v = xvalid_w2v.astype('float64')
xtrain_d2v = xtrain_d2v.astype('float64')
xvalid_d2v = xvalid_d2v.astype('float64')
test_bow = test_bow.astype('float64')
test_tfidf = test_tfidf.astype('float64')
test_w2v = test_w2v.astype('float64')
test_d2v = test_d2v.astype('float64')

#Preapring training dataset for lgb model
lgb_dtrain_bow = lgb.Dataset(xtrain_bow, label= ytrain)
#Training the lgb model with 100 iterations and computing training time
start = datetime.now()
lgb_cl_bow=lgb.train(lgb_params,lgb_dtrain_bow,200)
stop = datetime.now()
lgb_bow_exec_time = stop - start

#prediction on the validation set from Training dataset
prediction = lgb_cl_bow.predict(xvalid_bow)

    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction)>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_lgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING LIGHT GRADIENT BOOSTING\n")
print(bow_lgb_pred_score)

#Testing the built LIGHT GRADIENT BOOSTING model on test data

#predicting on test data
bow_lgb_test_pred = lgb_cl_bow.predict(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_lgb_test_pred_int = (bow_lgb_test_pred)>=0.3
bow_lgb_test_pred_int = bow_lgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_lgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('lgb_bow.csv', index=False)





#TF-IDF FEATURES - LIGHT GRADIENT BOOSTING model

#Preapring training dataset for lgb model
lgb_dtrain_tfidf = lgb.Dataset(xtrain_tfidf, label= ytrain)
#Training the lgb model with 100 iterations and computing training time
start = datetime.now()
lgb_cl_tfidf=lgb.train(lgb_params,lgb_dtrain_tfidf,100)
stop = datetime.now()
lgb_tfidf_exec_time = stop - start

 #prediction on the validation set from Training dataset
prediction = lgb_cl_tfidf.predict(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction)>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_lgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING LIGHT GRADIENT BOOSTING\n")
print(tfidf_lgb_pred_score)

#Testing the built LIGHT GRADIENT BOOSTING on test data

#predicting on test data
tfidf_lgb_test_pred = lgb_cl_tfidf.predict(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_lgb_test_pred_int = (tfidf_lgb_test_pred)>=0.3
tfidf_lgb_test_pred_int = tfidf_lgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_lgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('lgb_tfidf.csv', index=False)





#WORD2VEC FEATURES - LIGHT GRADIENT BOOSTING model

#Preapring training dataset for lgb model 
lgb_dtrain_w2v = lgb.Dataset(xtrain_w2v, label= ytrain)
#Training the lgb model with 100 iterations and computing training time
start = datetime.now()
lgb_cl_w2v=lgb.train(lgb_params,lgb_dtrain_w2v,100)
stop = datetime.now()
lgb_w2v_exec_time = stop - start

# prediction on the validation set from Training dataset
prediction = lgb_cl_w2v.predict(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction)>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_lgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING LIGHT GRADIENT BOOSTING\n")
print(w2v_lgb_pred_score)

#Testing the built LIGHT GRADIENT BOOSTING model on test data

#predicting on test data
w2v_lgb_test_pred = lgb_cl_w2v.predict(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_lgb_test_pred_int = (w2v_lgb_test_pred)>=0.3
w2v_lgb_test_pred_int = w2v_lgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_lgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('lgb_w2v.csv', index=False)





#DOC2VEC FEATURES - LIGHT GRADIENT BOOSTING model

#Preapring training dataset for lgb model
lgb_dtrain_d2v = lgb.Dataset(xtrain_d2v, label= ytrain)
#Training the lgb model with 100 iterations and computing training time
start = datetime.now()
lgb_cl_d2v=lgb.train(lgb_params,lgb_dtrain_d2v,100)
stop = datetime.now()
lgb_d2v_exec_time = stop - start

# prediction on the validation set from Training dataset
prediction = lgb_cl_d2v.predict(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction)>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_lgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING LIGHT GRADIENT BOOSTING\n")
print(d2v_lgb_pred_score)

#Testing the built LIGHT GRADIENT BOOSTING model on test data

#predicting on test data
d2v_lgb_test_pred = lgb_cl_d2v.predict(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_lgb_test_pred_int = (d2v_lgb_test_pred)>=0.3
d2v_lgb_test_pred_int = d2v_lgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_lgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('lgb_d2v.csv', index=False)


###################################################################################################################################################################


#Summary of F1 SCORES for all 5 models using 5 different algorithms
print("\n\nSummary of F1 SCORES for all 5 features using 5 different algorithms\n\n")

lreg_sum = ['   LOGREG ' , "bow: "+str(round(bow_lreg_pred_score,3)) , "tfidf: "+str(round(tfidf_lreg_pred_score,3)) ,
            "w2v: "+str(round(w2v_lreg_pred_score,3)) , "d2v: "+str(round(d2v_lreg_pred_score,3))]

svm_sum =  ['       SVM    ' , "\tbow: "+str(round(bow_svm_pred_score,3)) , "  tfidf: "+str(round(tfidf_svm_pred_score,3)) ,
           "\tw2v: "+str(round(w2v_svm_pred_score,3)) , "\td2v: "+str(round(d2v_svm_pred_score,3))]

rf_sum =   ['           RF     ', "\tbow: "+str(round(bow_rf_pred_score,3)) , "   tfidf: "+str(round(tfidf_rf_pred_score,3)) ,
          "\tw2v: "+str(round(w2v_rf_pred_score,3)) , "\td2v: "+str(round(d2v_rf_pred_score,3))]

xgb_sum =  ['       XGB    ' , "\tbow: "+str(round(bow_xgb_pred_score,3)) , "   tfidf: "+str(round(tfidf_xgb_pred_score,3)) ,
           "\tw2v: "+str(round(w2v_xgb_pred_score,3)) , "\td2v: "+str(round(d2v_xgb_pred_score,3))]

lgb_sum =  ['       LGB    ' , "\tbow: "+str(round(bow_lgb_pred_score,3)) , "   tfidf: "+str(round(tfidf_lgb_pred_score,3)) ,
           "\tw2v: "+str(round(w2v_lgb_pred_score,3)) , "\td2v: "+str(round(d2v_lgb_pred_score,3))]

for i,j,k,l,m in zip(lreg_sum , svm_sum , rf_sum , xgb_sum, lgb_sum):
    print(i,j,k,l,m)



#Summary of Training time for all 5 features using 5 different algorithms
print("\n\n\n\nSummary of training time for all 4 features using 4 different algorithms")

print("\n\nLOGISTRIC REGRESSION ALGORITHM TRAINING TIME")
print("\nbow:   "+str(lreg_bow_exec_time))
print("\ntfidf: "+str(lreg_tfidf_exec_time))
print("\nw2v:   "+str(lreg_w2v_exec_time))
print("\nd2v:   "+str(lreg_d2v_exec_time))

print("\n\nSUPPORT VECTOR MACHINE ALGORITHM TRAINING TIME")
print("\nbow:   "+str(svm_bow_exec_time))
print("\ntfidf: "+str(svm_tfidf_exec_time))
print("\nw2v:   "+str(svm_w2v_exec_time))
print("\nd2v:   "+str(svm_d2v_exec_time))

print("\n\nRANDOM FOREST ALGORITHM TRAINING TIME")
print("\nbow:   "+str(rf_bow_exec_time))
print("\ntfidf: "+str(rf_tfidf_exec_time))
print("\nw2v:   "+str(rf_w2v_exec_time))
print("\nd2v:   "+str(rf_d2v_exec_time))

print("\n\nEXTREME GRADIENT BOOSTING ALGORITHM TRAINING TIME")
print("\nbow:   "+str(xgb_bow_exec_time))
print("\ntfidf: "+str(xgb_tfidf_exec_time))
print("\nw2v:   "+str(xgb_w2v_exec_time))
print("\nd2v:   "+str(xgb_d2v_exec_time))

print("\n\nLIGHT GRADIENT BOOSTING ALGORITHM TRAINING TIME")
print("\nbow:   "+str(lgb_bow_exec_time))
print("\ntfidf: "+str(lgb_tfidf_exec_time))
print("\nw2v:   "+str(lgb_w2v_exec_time))
print("\nd2v:   "+str(lgb_d2v_exec_time))








