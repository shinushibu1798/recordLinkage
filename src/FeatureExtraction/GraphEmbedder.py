import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

class graphEmbedder:
    
    def __init__(self):
        self.entityNodes = set()
        self.Graphs = []
    
    def defineKeys(self,df,idCol):
        temp = set([str(x) for x in df[idCol]])
        self.entityNodes.update(temp)
    
    def embeddOrdinal(self,dfs,idCol,column,name,bins=10, equalBinSize = True):
        df = pd.concat([df[[idCol,column]] for df in dfs])
        if equalBinSize: temp = pd.qcut(df[column], q = bins)
        else: temp = temp = pd.cut(df[column], bins = bins)
        
        tempDf = df[[idCol]].copy(deep=True)
        tempDf['temp'] = temp
        graph = dict(tempDf.dropna().to_numpy())
        self.Graphs.append((name,graph))
        return True
    
    def embeddText(self,dfs,idCol, column,name, min_df = 0.01, method = 'BagOfWords'):
        df = pd.concat([df[[idCol,column]] for df in dfs])
        temp = df[[idCol, column]]
        temp = temp.dropna()
        if method == 'BagOfWords': CV = CountVectorizer(min_df=min_df, ngram_range = (1,1))
        
        output = CV.fit_transform(temp[column]).toarray()
        tokens = CV.get_feature_names()
        idVal = list(temp[idCol])
        
        graph = {}
        for i in range(output.shape[0]):
            words = []
            for j in range(output.shape[1]):
                if output[i][j] > 0:
                    words.append(tokens[j])
            graph[idVal[i]] = words
        
        self.Graphs.append((name,graph))
        return True
    
    def embeddCategorical(self,dfs,idCol,column,name):
        df = pd.concat([df[[idCol,column]] for df in dfs])
        graph = dict(df[[idCol,column]].to_numpy())
        self.Graphs.append((name,graph))
        return True
    
    def defineTruth(self, df):
        graph = dict(df.to_numpy())
        self.Graphs.append(('ground_truth',graph))
        return True
    
    def saveGraph(self, method = 'csv', fname = os.path.join(os.getcwd(),'heteroGraph.csv')):
        df = pd.DataFrame(columns = ['source', 'target','type'])
        for graph in self.Graphs:
            name = graph[0]
            links = graph[1]
            data = self._graphFixer(links)
            temp = pd.DataFrame(data = data, columns = ['source', 'target'])
            temp['type'] = name
            
            df = pd.concat([df, temp])
        temp = [str(x) for x in self.entityNodes]
        df_temp = pd.DataFrame(columns = ['source', 'target', 'type'])
        df_temp['source'] = temp
        df_temp['target'] = None
        df_temp['type'] = None
        df = pd.concat([df, df_temp])
        df.to_csv(os.path.join(os.getcwd(),fname), index = False)
    
    def _graphFixer(self, dictionary):
        temp = []
        for key, value in dictionary.items():
            if type(value)==list:
                for v in value:
                    temp.append((key,v))
            else:
                temp.append((key,value))
        return temp