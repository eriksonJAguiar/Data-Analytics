from pymongo import MongoClient
from datetime import datetime
import nltk
import pandas as pd
import numpy as np

#Faz a Normalização do texto            
def clean(dataframe):
	new_df = []
	for df in dataframe:
		#expr = re.sub(r"http\S+", "", df)
		expr = normalize('NFKD',df).encode('ASCII','ignore').decode('ASCII')
		filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"\w+") if not w in nltk.corpus.stopwords.words('portuguese')]
		frase = ""
		for f in filtrado:
			frase += f + " "
		new_df.append(frase)

	return new_df

def save(collection,text,date):
	collection.insert_one({'text':text,'date': date})

def init_mongo(dataBase,collection,connection):
	try:
		client = MongoClient(connection)

		db = client[dataBase]

		result = db[collection]

		return result

	except Exception as inst:
		print(type(inst))

def get_all_tweets(collection):
    return collection.find()

def get_n_tweets(collection,n):
    return collection.find().sort('text',1).limit(n)


if __name__ == '__main__':
    
    collection = init_mongo('bigData','tweets','localhost:27017')
    
    #save(collection,'Olá mundo',datetime.now().strftime("%d-%m-%y"))
    
    tweets = get_all_tweets(collection)
    
    for document in tweets:
        print("%s, %s"%(document['text'], document['date']))
    