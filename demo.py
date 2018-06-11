# -*- coding: latin-1 -*-
from __future__ import division
import os
from os import listdir
from os.path import isfile, isdir, join
import time
import re
import json
import math
import numpy as np
from random import shuffle
import pickle
import math
import string
import random
from types import *
import matplotlib.pyplot as plt
import collections
import pandas as pd
import nltk
import graphDoc
import sys
import getopt
import functions as functions

__author__ = 'Dimitrios Bountouridis'

def demo(input=""):
   
    print("Reading the documents (from json dataset)... input should be 'Data/jsonFile#jsonIndex")
    jsonFile = input.split("#")[0]
    jsonIndex = input.split("#")[1]
    cliqueOfArticles = functions.readJsonFile(jsonFile)
    contents = cliqueOfArticles[jsonIndex]["contents"]
    publications = cliqueOfArticles[jsonIndex]["publications"]
    titles = cliqueOfArticles[jsonIndex]["sentences"]
        
    print("Initialize object...")
    gDoc=graphDoc.graphDocuments(contents,publications,titles)   # initialize object with documents and publication classes e.g. cnn, fox
    
    print("Extracting sentence structure...")
    gDoc.sentenceProcess(withGA=True, output="temp/sentences.pkl")
    
    print("Computing sentence similarities...")
    gDoc.computeSentenceDistances(similarityFunction = "cosine")

    print("Keeping only the most important sentence-to-sentence similarities (thresholding)...")
    gDoc.reduceSentenceSimilarityFrame(pA=85,pB=93)

    print("Create graph... (no plotting)")
    gDoc.computeNetwork(plot=False,cliqueEdges=[])

    print("Clique finder in the graph...")
    gDoc.cliqueFinder(output="temp/cliquesFinal.json",orderby="median tf-idf score")


def main(argv):
    
    # Find cross-referenced pieces if information 
    # in the clique of documents with id 0.7653359236139566-5-6709
    demo("Data/dataset.json#0.7653359236139566-5-6709")
   
    
    


if __name__ == "__main__":
   main(sys.argv[1:])           
            

        
























