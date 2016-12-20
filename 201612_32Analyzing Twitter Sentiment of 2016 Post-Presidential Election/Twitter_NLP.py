#File: Twitter_NLP.py
# coding: utf-8

import re
import nltk
import random
import pickle
import pandas as pd
import numpy as np


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
#from nltk.corpus import wordnet
##from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.classify import ClassifierI
#from nltk.corpus import movie_reviews
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import word_tokenize
#from sklearn.svm import SVC, LinearSVC, NuSVC
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import preproc as pp

# create spark contexts
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)
remove_stops_udf_pos = udf(pp.remove_stops, StringType())
remove_features_udf_pos = udf(pp.remove_features, StringType())
tag_and_remove_udf_pos = udf(pp.tag_and_remove, StringType())
lemmatize_udf_pos = udf(pp.lemmatize, StringType())
check_blanks_udf_pos = udf(pp.check_blanks, StringType())

remove_stops_udf_neg = udf(pp.remove_stops, StringType())
remove_features_udf_neg = udf(pp.remove_features, StringType())
tag_and_remove_udf_neg = udf(pp.tag_and_remove, StringType())
lemmatize_udf_neg = udf(pp.lemmatize, StringType())
check_blanks_udf_neg = udf(pp.check_blanks, StringType())


# Load a text file and convert each line to a Row.
data_rdd_pos = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/positive.txt")
parts_rdd = data_rdd.map(lambda l: l.split("\t"))
# Filter bad rows out
garantee_col_rdd = parts_rdd.filter(lambda l: len(l) == 3)
typed_rdd = garantee_col_rdd.map(lambda p: (p[0], p[1], float(p[2])))
# Create DataFrame
data_df_pos = sqlContext.createDataFrame(typed_rdd)

lang_df_pos = data_df_pos.withColumn("lang", check_lang_udf(data_df["text"]))
en_df_pos = lang_df.filter(lang_df["lang"] == "en")
rm_stops_df_pos = en_df_pos.withColumn("stop_text", remove_stops_udf_pos(en_df["text"]))
rm_features_df_pos = rm_stops_df_pos.withColumn("feat_text", remove_features_udf_pos(rm_stops_df["stop_text"]))
tagged_df_pos = rm_features_df.withColumn("tagged_text", tag_and_remove_udf_pos(rm_features_df["feat_text"]))
lemm_df_pos = tagged_df_pos.withColumn("lemm_text", lemmatize_udf(tagged_df["tagged_text"]))
check_blanks_df_pos = lemm_df_pos.withColumn("is_blank", check_blanks_udf_pos(lemm_df["lemm_text"]))
no_blanks_df = check_blanks_df_pos.filter(check_blanks_df["is_blank"] == "False")
no_blanks_df_pos.withColumnRenamed(no_blanks_df["lemm_text"], "text")
dedup_df_pos = no_blanks_df_pos.dropDuplicates(['text', 'label'])
data_set = dedup_df_pos
splits = data_set_pos.randomSplit([0.6, 0.4])

data_rdd_neg = sc.textFile("/Users/ShihaoZhang/Desktop/Big_Data/Project/negative.txt")
parts_rdd= data_rdd_neg.map(lambda l: l.split("\t"))
# Filter bad rows out
garantee_col_rdd = parts_rdd.filter(lambda l: len(l) == 3)
typed_rdd = garantee_col_rdd.map(lambda p: (p[0], p[1], float(p[2])))
# Create DataFrame
data_df_neg = sqlContext.createDataFrame(typed_rdd)

lang_df = data_df.withColumn("lang", check_lang_udf(data_df["text"]))
en_df = lang_df.filter(lang_df["lang"] == "en")
rm_stops_df_neg = en_df.withColumn("stop_text", remove_stops_udf_neg(en_df["text"]))
rm_features_df_neg = rm_stops_df.withColumn("feat_text", remove_features_udf_neg(rm_stops_df_neg["stop_text"]))
tagged_df_neg = rm_features_df.withColumn("tagged_text", tag_and_remove_udf_pos(rm_features_df_neg["feat_text"]))
lemm_df_neg = tagged_df.withColumn("lemm_text", lemmatize_udf(tagged_df_neg["tagged_text"]))
check_blanks_df_neg = lemm_df.withColumn("is_blank", check_blanks_udf_neg(lemm_df["lemm_text"]))
no_blanks_df_neg = check_blanks_df_neg.filter(check_blanks_df["is_blank"] == "False")
no_blanks_df_neg.withColumnRenamed(no_blanks_df["lemm_text"], "text")
dedup_df_neg = no_blanks_df_neg.dropDuplicates(['text', 'label'])
data_set_neg = dedup_df_neg
splits_neg = data_set_neg.randomSplit([0.6, 0.4])


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
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
prediction_df = result.select("text", "label", "prediction")

datasci_df = prediction_df.filter(prediction_df['label']==0.0)
datasci_df.show(truncate=False)

ao_df = prediction_df.filter(prediction_df['label']==1.0)
ao_df.show(truncate=False)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(result, {evaluator.metricName: "precision"})
#Prepare the txt files
#dataset = pd.read_csv("/Users/ShihaoZhang/Desktop/Big_Data/Project/Sentiment_Analysis_Dataset.csv")
# '1' for positive sentiment and '0' for negative sentiment
#neg = dataset[dataset['Sentiment'] == 0]['SentimentText']
#pos = dataset[dataset['Sentiment'] == 1]['SentimentText']
#np.savetxt(r'/Users/ShihaoZhang/Desktop/Big_Data/Project/neg.txt', neg.values, fmt='%s')
#np.savetxt(r'/Users/ShihaoZhang/Desktop/Big_Data/Project/pos.txt', pos.values, fmt='%s')

#short_pos = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/positive.txt","r").read()
#short_neg = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/negative.txt","r").read()

# read 'pos' and 'neg' data into 'documents'
documents = []
for r in short_pos.split('\n'):
    documents.append( (r, "pos") )
for r in short_neg.split('\n'):
    documents.append( (r, "neg") )
#save document into pickled_algos
save_documents = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1", s)
#end



#start find_features
def find_features(document):
    #decode with 'utf-8'
    words = word_tokenize(document.decode('utf-8'))
    #remove the word if it not begin with alphabet
    words = [w for w in words if not re.match("^[a-zA-Z]+.*", w) is None]
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
#end

featuresets = [(find_features(rev), category) for (rev, category) in documents]
#save features
save_featuresets = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
training_number = int(len(featuresets)*0.8)
training_set = featuresets[:training_number]
testing_set = featuresets[training_number:]

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

#Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#MNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

#Bernouli_NB Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

#Logistic Regression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

#Linear SVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

#SGDC Classifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)
#save it into pickled_algos
save_classifier = open("/Users/ShihaoZhang/Desktop/Big_Data/Project/pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()
