#coletar dados do twitter utilizando o streaming com a api

import tweepy
from datetime import datetime
import csv
from unicodedata import normalize
import pandas as pd
import nltk
import pymongo
import re
import json

#Credencias de acesso App Twitter
consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"

#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication)


def write_dataframe(df,file):
    df.to_csv('%s.csv'%file, mode='a', sep=';',index=False, header=False)

def init_mongo(dataBase,collection,connection):
	try:
		client = MongoClient(connection)

		db = client[dataBase]

		result = db[collection]

		return result

	except Exception as inst:
		print(type(inst))    

def save(collection,record):
    collection.insert_one(record)

#Faz a Normalização do texto            
def clear(dataframe):
    new_df = []
    for expressao in dataframe:
        expr = re.sub(r"http\S+", "", expressao)
        #expr = re.sub(r"[@#]\S+","",expr)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('portuguese')]
        frase = ""
        for f in filtrado:
            frase += f + " "
        
        new_df.append(frase)
    
    return new_df

#coleta os tweets pela tag e retorna o tweet e a data da coleta
def get_tweets(tags):
    tweets = []
    day = []
    for tg in tags:
        results = api.search(q=tg)
        for r in results:
            tweets.append(r.text)
            day.append(datetime.now().strftime("%d-%m-%y"))
            print('Tweet: %s, Data: %s'%(r.text,datetime.now().strftime("%d-%m-%y")))
    
    return tweets,day

#coleta as top trends
def get_trends():
    tags = []
    trends = api.trends_place(23424768)
    data = trends[0]
    trend = data['trends']
    for item in trend:
        tag = str(item['name'])
        line  = (str(datetime.now()),tag)
        tags.append(line)
        
    
    df_trends = pd.DataFrame(tags,columns=['date','trend'])

    return df_trends

#set 1 save file .csv or set 2 save mongo collection
def record_data(i,tweets,trends):
    if i == 1:
        write_dataframe(trends,'trends')
        write_dataframe(tweets,'tweets')
    
    elif i == 2:
        coll = init_mongo('bigData','tweets','localhost:27017')
        records_trends = json.loads(trends.T.to_json()).values()
        records_tweets = json.loads(tweets.T.to_json()).values()
        print()
        save(coll,records_trends)
        save(coll,records_tweets)

            

if __name__ == '__main__':
    
    tags = []
    line = []
    df = pd.DataFrame()
    df_trend = get_trends()
    print(df_trend)
    
    df['tweets'], df['day'] = get_tweets(df_trend['trend'])
    df['tweets'] = clear(df['tweets'])
    print(df)

    record_data(1,df,df_trend)
    