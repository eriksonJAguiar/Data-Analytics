import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

def read_csv(file):
    df = pd.DataFrame.from_csv('%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')
    df = df.reset_index()
    
    return df
    
def convert_dataset(df):
    
    label_enco = LabelEncoder()
    
    
    df['País'] = label_enco.fit_transform(df['País'])
    df['Tipo-viagem'] = label_enco.fit_transform(df['Tipo-viagem'])
    df['Piscina'] = label_enco.fit_transform(df['Piscina'])
    df['Academia'] = label_enco.fit_transform(df['Academia'])
    df['Tenis'] = label_enco.fit_transform(df['Tenis'])
    df['Spa'] = label_enco.fit_transform(df['Spa'])
    df['Casino'] = label_enco.fit_transform(df['Casino'])
    df['Internet'] = label_enco.fit_transform(df['Internet'])
    df['Hotel'] = label_enco.fit_transform(df['Hotel'])
    
    
    return df

def convert_input(value):
     
    label_enco = LabelEncoder()
    
    #label_enco.fit_transform()

def inverse_result(result):
    
    label_enco = LabelEncoder()
    
    label_enco.fit(['Circus Circus Hotel & Casino Las Vegas','Excalibur Hotel & Casino','Monte Carlo Resort&Casino','Treasure Island- TI Hotel & Casino',
                    'Tropicana Las Vegas - A Double Tree by Hilton Hotel','Caesars Palace','The Cosmopolitan Las Vegas','The Palazzo Resort Hotel Casino',
                   'Wynn Las Vegas','Trump International Hotel Las Vegas','The Cromwell','Encore at wynn Las Vegas','Hilton Grand Vacations on the Boulevard',
                   "Marriott's Grand Chateau",'Tuscany Las Vegas Suites & Casino','Hilton Grand Vacations at the Flamingo','Wyndham Grand Desert',
                   'The Venetian Las Vegas Hotel','Bellagio Las Vegas','Paris Las Vegas','The Westin las Vegas Hotel Casino & Spa'])
    
    r = label_enco.inverse_transform(result)
    
    return r

def predict_tree(X,y,input_):
    
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
    
    pred = clf.fit(X,y).predict(input_)
    
    return inverse_result(pred)

def split_dataset(df):
    
    new_df = pd.DataFrame()
    
    new_df['País'] = df['País']
    new_df['Reviews'] = df['Reviews']
    new_df['Hotel-reviews'] = df['Hotel-reviews']
    new_df['Votes'] = df['Votes']
    new_df['Score'] = df['Score']
    new_df['Tipo-viagem'] = df['Tipo-viagem']
    new_df['Piscina'] = df['Piscina']
    new_df['Academia'] = df['Academia']
    new_df['Tenis'] = df['Tenis']
    new_df['Spa'] = df['Spa']
    new_df['Casino'] = df['Casino']
    new_df['Internet'] = df['Internet']
    new_df['Estrelas'] = df['Estrelas']
    
    train = new_df.values
    
    target = df['Hotel'].values
    
    return train,target

def main():
    df = read_csv('hotel')
    df = convert_dataset(df)
    train,target = split_dataset(df)
    pred = predict_tree(train,target,[1,36,5,20,4,1,1,0,1,1,0,0,4])
    
    print(pred)


if __name__ == '__main__':
    
    main()
    