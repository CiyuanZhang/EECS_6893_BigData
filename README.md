# EECS_6893_BigDataBig data analytics_CU ELEN6893
Final Project
%Analyzing Twitter Sentiment of 2016 Post-Presidential Election

ProjectID: 
			%201612-32
Team Member:
			Lingqing XU        (Lx2201)
			Ciyuan ZHANG   (cz2350)
			Shihao ZHANG   (sz2531)

Data range: 
			11/14/2016 to 12/14/2016
Keywords: 
		@hillaryclinton 
		@RealDonaldTrump
		#hillaryclinton 
		#donaldtrump 
		Clinton 
		Hillary 
		Donald 
		Trump

Raw tweets (1.17GB) + Processed Data + Final result (html graph) :
Google Drive Shared Link:
https://drive.google.com/drive/folders/0B1ScL3U5Lv3oOEVtSWRQcHBIN28?usp=sharing
                
%RUNNING:

ENVIRONMENT: 
		python 3.5.2, spark 2.0.1

AUTHENTICATION:
#https://apps.twitter.com/app/13197060/show
#https://twitter.com/?lang=en
#Owner: XuLingqing
#Owner ID: 774060576100192256
API_KEY(Consumer Key ) = 'wO3e5cZTaP9MuSIDc58RBz6YB'
API_SECRET(Consumer Secret) = 'rUF046vyBebhlz344dJnB20xHN29RlsISUxRkX4AXtVgkpwOie'
ACCESS_TOKEN = '774060576100192256-vVFgqXwzqTS5nWMT5N3Dyzo34t0asMg'
ACCESS_TOKEN_SECRET = '9Xf1DyzyRiLAzcEHOi1rDcijsG3EQ57UnVfLqphKPkIKb'

twitter201612_32.py : Used to pull and write data to data.txt file. In the txt documents, each line is tweets of a specific user that contains the keyword. We stream tweets in real time using [Twitter's API](https://dev.twitter.com/streaming/public) and tweepy(https://github.com/tweepy/tweepy). These tweets are filtered on terms related to the two candidates, and formatted with the tweets containing candidate name. 

TwitterNLP : Used to process data

Result: 
We will get the percentage of tweets that use positive words, or rather hold positive attitude towards


 
