#File: sentiment_mod.py

from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
#from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import ElementwiseProduct
from pyspark.mllib.linalg import Vectors
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from statistics import mode
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import preproc as pp
'''
import re
import csv
import nltk
import random
import pickle
import pandas as pd
import numpy as np
from statistics import mode
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import ClassifierI
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
'''

#sc = pyspark.SparkContext()
#sqlContext = SQLContext(sc)
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)
remove_stops_udf = udf(pp.remove_stops, StringType())
remove_features_udf= udf(pp.remove_features, StringType())
tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
lemmatize_udf = udf(pp.lemmatize, StringType())
check_blanks_udf = udf(pp.check_blanks, StringType())

data_rdd = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 1 .txt")
parts_rdd = data_rdd.map(lambda l: l.split("\t"))
# Filter bad rows out
garantee_col_rdd = parts_rdd.filter(lambda l: len(l) == 3)
typed_rdd = garantee_col_rdd.map(lambda p: (p[0], p[1], float(p[2])))
# Create DataFrame
data_df = sqlContext.createDataFrame(typed_rdd, ["text", "id", "label"])
lang_df = data_df.withColumn("lang", check_lang_udf(data_df["text"]))
en_df = lang_df.filter(lang_df["lang"] == "en")

rm_stops_df = en_df.withColumn("stop_text", remove_stops_udf(en_df["text"]))
rm_features_df = rm_stops_df.withColumn("feat_text", remove_features_udf(rm_stops_df["stop_text"]))
tagged_df = rm_features_df.withColumn("tagged_text", tag_and_remove_udf(rm_features_df["feat_text"]))
lemm_df = tagged_df.withColumn("lemm_text", lemmatize_udf(tagged_df["tagged_text"]))

check_blanks_df = lemm_df.withColumn("is_blank", check_blanks_udf(lemm_df["lemm_text"]))
no_blanks_df = check_blanks_df.filter(check_blanks_df["is_blank"] == "False")



# Load already cleaned data
def reload_checkpoint(data_rdd):
    parts_rdd = data_rdd.map(lambda l: l.split("\t"))
    # Filter bad rows out
    garantee_col_rdd = parts_rdd.filter(lambda l: len(l) == 3)
    typed_rdd = garantee_col_rdd.map(lambda p: (p[0], p[1], float(p[2])))
    # Create DataFrame
    df = sqlContext.createDataFrame(typed_rdd)
    return df
test_df = reload_checkpoint(test_rdd)

data_rdd_2 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 2 .txt")
data_rdd_3 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 3 .txt")
data_rdd_4 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 4 .txt")
data_rdd_5 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 5 .txt")
data_rdd_6 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 6 .txt")
data_rdd_7 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 7 .txt")
data_rdd_8 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 8 .txt")
data_rdd_9 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 9 .txt")
data_rdd_10 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 10 .txt")
data_rdd_11 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 11 .txt")
data_rdd_12 = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential/data/raw/pelection_ 12 .txt")


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#documents_f = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/documents.pickle", "rb")
#documents = pickle.load(documents_f)
#documents_f.close()

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1", s)
#end

n = 10000
def find_features(document):
    #decode with 'utf-8'
    words = word_tokenize(document.decode('utf-8'))
    #remove the word if it not begin with alphabet
    words = [w for w in words if not re.match("^[a-zA-Z0-9]+.*", w) is None]
    #remove punctuation
    words = [w.strip('#?:"*&|~!@$-_+/\[]{};^<>`()_,.') for w in words]
    #replaceTwoOrMore, eg: 'juuuuuust' --> 'just'
    words = [replaceTwoOrMore(w) for w in words]
    #stemming the words
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    #remove the "stop words"
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    temp = nltk.FreqDist(words).most_common(n)
    #word features and then save it to pickle
    word_features = []
    for w in temp:
        word_features.append(w[0])
    #save_word_features = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/word_features5k.pickle","wb")
    #pickle.dump(word_features, save_word_features)
    #save_word_features.close()
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
idf = IDF(minDocFreq=3, inputCol="features", outputCol="idf")
nb = NaiveBayes()
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])


paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 1.0]).build()


cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=paramGrid,
                    evaluator=MulticlassClassificationEvaluator(),
                    numFolds=4)

cvModel = cv.fit(training_df)
result = cvModel.transform(test_df)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(result, {evaluator.metricName: "precision"})
#featuresets_f = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/featuresets.pickle", "rb")
#featuresets = pickle.load(featuresets_f)
#featuresets_f.close()

#prepare tge training data and testing data
#random.shuffle(featuresets)
#training_number = int(len(featuresets)*0.8)
#training_set = featuresets[:training_number]
#testing_set = featuresets[training_number:]

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

#test the classifier:
print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

'''
temp = open("/Users/ShihaoZhang/Downloads/data_Hillary2.txt","r").read()

documents = []
for r in temp.split('\n'):
    documents.append(r)

result = []
for text in documents:
    w = sentiment(text)
    print(w)
    result.append(w)

#save_results = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/temp/result_1.pickle","wb")
#pickle.dump(featuresets, save_results)
#save_results.close()

result_csv = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/Twitter-Presidential-Race-Sentiment-Clustering/data/result/output_Hillary2.csv",'wb')

wr = csv.writer(result_csv, quoting=csv.QUOTE_ALL)
wr.writerow(result)

result_csv.close()
'''
