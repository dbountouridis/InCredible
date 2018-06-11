from __future__ import division
import os
from os import listdir
from os.path import isfile, isdir, join
import time
import json
import numpy as np
import pickle
import string
import matplotlib.pyplot as plt
import collections
import pandas as pd
import nltk
from nltk.collocations import *
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import seaborn as sns
import networkx as nx
import itertools
import gaOnSentences as ga
import functions as functions
from operator import itemgetter


tfVectorizer = TfidfVectorizer(analyzer='word', stop_words='english',lowercase=True)
countVectorizer = CountVectorizer(stop_words='english', lowercase=True)


# the main class for holding a set of documents, parsing then into sentences, computing their similarity, generaring a graph and finding its cliques
class graphDocuments(object):
	def __init__(self, documents,publications,titles):
		
		self.N = len(documents)
		self.DocumentsText = []
		self.publications = publications
		self.titles = titles
		self.SentenceSimilarityFrame = [] 			# dataframe holding sentence-to-sentence similarities
		self.ReducedSentenceSimilarityFrame = [] 	# reduced sentence-to-sentence similarities
		self.G = [] 								# sentence network
		self.Cliques = [] 							# cliques in the network
		self.cliqueEdges = [] 						# list of edges in all cliques
		self.tfD = []      							# store the parsed, lemmatized words for tf-idf later
		
		for content in documents:
			self.DocumentsText.append(content)
			data = " ".join(functions.parseText(content))
			self.tfD.append(data)  


	# process documents into sentences structure
	def sentenceProcess(self, withGA=True, output="temp/sentences.pkl"):
		
	    tfFrame = self.tfIdfForDocuments(self.tfD)  # tf-idf

	    # create dict structure SentencesFrame to hold the sentence features
	    S_ = []
	    for i, document in enumerate(self.DocumentsText):

	    	# some initial cleaning
	        t1 = ["U.S.", "U. S. A.", "U.S.A.", "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", ".”"," .”", "U.N."]+[c+"." for c in string.ascii_uppercase]
	        t2 = ["US", "USA", "USA", "Mr", "Mrs", "Ms", "Dr", "Jr","”.", " ”.","UN"]+list(string.ascii_uppercase)
	        for j, word1 in enumerate(t1): document = document.replace(word1, t2[j])
	        sentences=functions.sentenceTokenize(document)

	        # genetic algorithm for concatenating sentences
	        if withGA:
		        sentences=functions.sentenceTokenize(document)[:-1]
		        s = [np.sum(np.array(tfFrame.loc[tfFrame['feature'].isin(list(
		            set(functions.parseText(s))))]["Mean tf.idf"])) for j, s in enumerate(sentences)]
		        top = ga.GAonSequences(s)
		        sentences = ga.seriesSentence(sentences, top)

		    # save structure
	        for j, s in enumerate(sentences):
	            if len(s.split(" ")) > 2: 	# if sentence is of size 2 at least
	                S_.append([str(i)+"-"+functions.simpleCleanText(s)[0:np.min([20, len(functions.simpleCleanText(s))])], s, list(set(functions.parseText(
	                    s))), np.sum(np.array(tfFrame.loc[tfFrame['feature'].isin(list(set(functions.parseText(s))))]["Mean tf.idf"])), j, j/len(sentences), i, self.publications[i]])

	    # export to dataframe, pickle
	    self.SentencesFrame = pd.DataFrame(data=S_, columns=[ "node id", "raw sentence", "words", "tf-idf score", "sentence number", "sentence position", "class", "publication"])
	    self.SentencesFrame.to_pickle(output)


	# compute the distances/similarities between sentences
	def computeSentenceDistances(self,similarityFunction = "cosine"):

		if similarityFunction == "cosine":
		    print("(the scikitlearn package)")
		    tf = TfidfVectorizer(analyzer = 'word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
		    tfidf_matrix = TfidfVectorizer().fit_transform(np.array(self.SentencesFrame["raw sentence"]))
		    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

		D_ = []  # D holds the sentence distance data, later stored as data frame
		for i in range(cosine_similarities.shape[0]):
		    row1 = self.SentencesFrame.iloc[i] 
		    for j in range(cosine_similarities.shape[0]):
		        row2 = self.SentencesFrame.iloc[j] 
		        if i>j and row1["class"]!=row2["class"]:
		            if cosine_similarities[i][j]>0 and row1["raw sentence"]!=row2["raw sentence"]:
		                D_.append([row1["node id"],row2["node id"],float(cosine_similarities[i][j]),row1["tf-idf score"],row2["tf-idf score"],row1["class"],row2["class"],row1["raw sentence"],row2["raw sentence"],1-np.mean([row1["sentence position"],row2["sentence position"]])])
		                D_.append([row2["node id"],row1["node id"],float(cosine_similarities[i][j]),row2["tf-idf score"],row1["tf-idf score"],row2["class"],row1["class"],row2["raw sentence"],row1["raw sentence"], 1-np.mean([row1["sentence position"],row2["sentence position"]])])
		# to dataframe
		self.SentenceSimilarityFrame = pd.DataFrame(data=D_,columns=['NodeA',"NodeB","Similarity","Aimportance","Bimportance","DocumentA","DocumentB","SentenceA","SentenceB","Mean position from end"]) 
		self.SentenceSimilarityFrame=self.SentenceSimilarityFrame.drop_duplicates()     # drop duplicates


	# reduce the sentence-to-sentence connections
	def reduceSentenceSimilarityFrame(self,pA = 84,pB = 90,output = "temp/reduceddata.pkl"):
		# reducing the edges, keeping only the one most similar edge per class to a node
	    self.SentenceSimilarityFrame = self.SentenceSimilarityFrame.sort_values(by=['Aimportance',"Bimportance"],ascending=False)  # sort by tf-idf importance
	    
	    # we reduce sentence connectios based on the overall similarity distribution
	    self.percentileA = np.percentile(np.array(self.SentenceSimilarityFrame["Similarity"]).astype(float),pA)
	    self.percentileB = np.percentile(np.array(self.SentenceSimilarityFrame["Similarity"]).astype(float),pB)
	    print("Percentiles used for filtering:",self.percentileA,self.percentileB)
	    
	    # reduce connections
	    counter = 0
	    for nodeA,group in self.SentenceSimilarityFrame.groupby("NodeA"):
	        counterNode = 0
	        for w,group2 in group.groupby("DocumentB"):
	            sortedGroup = group2.sort_values(by=['Similarity',"Bimportance"],ascending=False) 
	            selected = np.array(sortedGroup.index)[:1] # the indeces of the most
	            bestSimilarity = np.array(sortedGroup["Similarity"])[0]
	            sortedGroup = sortedGroup.drop([index for index in np.array(sortedGroup.index)[:] if index not in selected]) # drop the non important
	            if bestSimilarity>self.percentileA: # keep only high similar edges
		            if counterNode==0:
		            	reducedNodeData = sortedGroup.copy()
		            else:
		            	reducedNodeData = reducedNodeData.append(sortedGroup)
		            counterNode+=1  
	        if counter==0:
	            self.ReducedSentenceSimilarityFrame = reducedNodeData.copy()
	        else:
	            self.ReducedSentenceSimilarityFrame = self.ReducedSentenceSimilarityFrame.append(reducedNodeData)
	        counter+=1
 
	    self.ReducedSentenceSimilarityFrame.to_pickle(output)
	    


	# create a networkx graph
	def computeNetwork(self,plot=False,cliqueEdges=[]):
	    
	    data = self.ReducedSentenceSimilarityFrame.copy()
	    data = data.as_matrix()
	    self.G = nx.Graph()
	    
	    # nodes 
	    nodes = list(set(np.array(data)[:,0]).intersection(set(np.array(data)[:,1])))
	    for node in nodes: self.G.add_node(node)

	    # nodes and their size
	    nodesAndSize = {u:float(v) for (u,v) in zip(np.array(data)[:,0].tolist()+np.array(data)[:,1].tolist(),np.array(data)[:,3].tolist()+np.array(data)[:,4].tolist()) }

	    # first remove some worthless tupless
	    if len(cliqueEdges)>0 and onlycliqueEdges: 
	        for d in data:  
	            if (d[0],d[1]) in cliqueEdges:
	                self.G.add_edge(d[0], d[1],weight=d[2])
	    else:
	        for d in data: self.G.add_edge(d[0], d[1],weight=d[2]) 
	       
	    # simple graph
	    if plot:
	        # drawing
	        sns.set_context("notebook", font_scale=0.8, rc={"lines.linewidth": 1.0})
	        sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
	        colorsN = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
	        colors = sns.xkcd_palette(colorsN)
	        colors += sns.color_palette("Set2", 10)
	        plt.subplot(111)

	        # layout
	        pos = nx.spring_layout(self.G,iterations=30) 

	        # color and draw nodes
	        color_map = []
	        for node in self.G: 
	            color=colors[int(node.split("-")[0])]
	            color_map.append(color)
	            nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color = color,node_size=200*nodesAndSize[node],node_shape="o") 

	        # edges
	        for (u, v, d) in self.G.edges(data=True):
	            edge=(u,v)
	            edgeAlt=(v,u)
	            weight=d['weight']
	            #if weight<percentile50: weight=0
	            edge_color_="black"       
	            if edge in cliqueEdges or edgeAlt in cliqueEdges: 
	                edge_color_="red"
	                weight=1
	            nx.draw_networkx_edges(self.G, pos, edge_color=edge_color_, edgelist=[edge,edgeAlt], width=np.power(3,weight),alpha=np.min([1,weight])) 
	        
	        # labels
	        nx.draw_networkx_labels(self.G, pos, font_size=6, font_family='sans-serif')
	        
	        # plot
	        if plot: 
	            plt.show()
	            plt.close() 
	

	# find cliques in the graph
	def cliqueFinder(self,output = "temp/cliquesFinal.json", orderby = "average clique similarity"):
	    cliques=nx.find_cliques(self.G)
	    self.Cliques={}
	    D_=self.ReducedSentenceSimilarityFrame.copy()
	
	    for cliqueIndex,clique in enumerate(cliques):
	        classes = list(set([item.split("-")[0] for item in clique]))
	        pubs = [self.publications[int(item.split("-")[0])] for item in clique]
	        items = list(set([item for item in clique]))
	        text = [np.array(D_.loc[D_['NodeA'] == node]["SentenceA"])[0] for node in items]
	        importance = [np.array(D_.loc[D_['NodeA'] == node]["Aimportance"])[0] for node in items]
	        
	        # skip if the clique contains double classes
	        if len(items)!=len(classes) or len(items)<=2: continue   

	        M = np.zeros([len(items),len(items)])   # numpy matrix to hold distances in clique
	        for i,itemA in enumerate(items): 
	            row = D_[D_['NodeA'] == itemA]
	            for j,itemB in enumerate(items):
	                if j!=i:
	                    row2 = row.loc[row['NodeB'] == itemB]
	                    if len(np.array(row2["Similarity"]))>0:
	                        M[i,j]=np.array(row2["Similarity"])[0]
	                        M[j,i]=M[i,j]
	        iu2 = np.triu_indices(len(items), k=1)  # upper triangle of the matrix only (remove diagonal)
	        meanSimilarity = np.mean(M[iu2].flatten())
	        
	        # if every edge with similarity higher than a threshold, store it
	        if sum(M[iu2].flatten()<self.percentileB)==0:    
	            self.Cliques.update({str(meanSimilarity)+"-"+str(len(items))+"-"+str(cliqueIndex):{"clique":items,"average clique similarity":meanSimilarity,"sentences":text,"tf-idf score":importance,"median tf-idf score":np.median(importance),"publications":pubs}})
	        
	        # if not, start removing nodes one by one until their similirity is higher than threshold
	        elif sum(M[iu2].flatten()<self.percentileB)>0 and len(classes)>3:   
	            s=np.argsort(np.mean(M,axis=0))     # sort nodes by average lowest similarity
	            for i in range(len(s)):
	                N=M.copy()
	                N=np.delete(N,s[:i],axis=0)
	                N=np.delete(N,s[:i],axis=1)
	                iu3 = np.triu_indices(N.shape[0], k=1)
	                if sum(N[iu3].flatten()<self.percentileB)==0:    # we found which nodes to remove
	                    nodesToRemove=s[:i]
	                    break
	            M=np.delete(M,nodesToRemove,axis=0) 
	            M=np.delete(M,nodesToRemove,axis=1)
	            items=np.delete(np.array(items),nodesToRemove).tolist()
	            pubs = [self.publications[int(item.split("-")[0])] for item in items]
	            text=np.delete(np.array(text),nodesToRemove).tolist()
	            importance=np.delete(np.array(importance),nodesToRemove).tolist()
	            iu2 = np.triu_indices(len(items), k=1)
	            meanSimilarity = np.mean(M[iu2].flatten())
	            
	            # store it
	            if len(items)>=3:
	                self.Cliques.update({str(meanSimilarity)+"-"+str(len(items))+"-"+str(cliqueIndex):{"clique":items,"average clique similarity":meanSimilarity,"sentences":text,"tf-idf score":importance,"median tf-idf score":np.median(importance),"publications":pubs}})

	    # remove overlapping cliques
	    sortedKeys = np.array(list(self.Cliques.keys()))[np.argsort([self.Cliques[key][orderby] for key in self.Cliques.keys()])[::-1]]
	    itemholder=[]

	    for key in sortedKeys:    # sort by score computed previously
	        f = self.Cliques[key]
	        clique = f["clique"]
	        items = list(set([item for item in clique]))
	        if len(itemholder)==0:
	            itemholder+=items
	        else:
	        	amountOfOverlappingItems = sum([item in itemholder for item in items])
	        	if amountOfOverlappingItems > 0: # if a node appears already in a more important clique
	        		#print("Found overlapping", sum([item in itemholder for item in items]), "out of ",len(items))
	        		if len(items) - amountOfOverlappingItems<3:
	        			#print("Removing overlapping items would result to small clique, -> remove clique.")
	        			self.Cliques.pop(key)
	        		else:
	        			#print("remove items",[item in itemholder for item in items])
	        			text = [np.array(D_.loc[D_['NodeA'] == node]["SentenceA"])[0] for node in items]
	        			
	        			nodesToRemove=[]
	        			for it in [item for item in items if item in itemholder]: nodesToRemove.append(items.index(it))
	        			
	        			items = np.delete(np.array(items),nodesToRemove).tolist()
	        			pubs = [self.publications[int(item.split("-")[0])] for item in items]
	        			text = np.delete(np.array(text),nodesToRemove).tolist()
	        			importance = [np.array(D_.loc[D_['NodeA'] == node]["Aimportance"])[0] for node in items]
	        			meanSimilarity = f["average clique similarity"] # we keep it the same for now
	        			self.Cliques.pop(key)
	        			self.Cliques.update({str(meanSimilarity)+"-"+str(len(items))+"-"+str(cliqueIndex):{"clique":items,"average clique similarity":meanSimilarity,"sentences":text,"tf-idf score":importance,"median tf-idf score":np.median(importance),"publications":pubs}})
	        			itemholder+=items
	        	else:
	        		itemholder+=items

	    # export final set of cliques
	    with open(output, 'w') as f:
	        json.dump(self.Cliques, f, sort_keys=True,indent=4, separators=(',', ': '))

	    # compute all the clique edges
	    self.cliqueEdges = []
	    for keyIndex in self.Cliques.keys():
	        self.cliqueEdges += list(itertools.permutations(self.Cliques[keyIndex]["clique"],2))
	    self.cliqueEdges = list(set(self.cliqueEdges))


	# top n tfidf features
	def top_tfidf_feats(self,row, features, top_n=25):
	    # Get top n tfidf values in row and return them with their corresponding feature names
	    topn_ids = np.argsort(row)[::-1][:top_n]
	    top_feats = [(features[i], row[i]) for i in topn_ids]
	    df = pd.DataFrame(top_feats)
	    df.columns = ['feature', 'tfidf']
	    return df


	# compute a  dataframe holding the tf-idf values of a group of lemmatized,parsed documents
	def tfIdfForDocuments(self,D):
	    xtrain = tfVectorizer.fit_transform(D)
	    features = tfVectorizer.get_feature_names()
	    
	    for row_id in range(len(D)):
	        row = np.squeeze(xtrain[row_id].toarray()) # sparce matrix to array
	        df=self.top_tfidf_feats(row, features,top_n=len(features)) # 48 words with highest tfidf in pandas array
	        df.columns = ['feature', 'tf.idf'+str(row_id)]
	        if row_id==0:
	            B=df.copy()
	        else:
	            B=pd.merge(B,df,on=["feature"])
	        B['Mean tf.idf'] = B.mean(axis=1)
	        B['Product tf.idf'] = B.product(axis=1)
	        B=B.sort_values(by=['Product tf.idf','Mean tf.idf'],ascending=False)

	    return B 
	    
	




