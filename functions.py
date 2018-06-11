from __future__ import division
import os
from os import listdir
import string
import inflect
import nltk
from nltk.collocations import *
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json

PUNCTUATION = string.punctuation
NORMALSYMBOLS = list(string.ascii_uppercase + string.ascii_lowercase+string.digits)
wordnet_lemmatizer = WordNetLemmatizer()
inflectEngine = inflect.engine()

# parse and lemmatize a string

def sentenceTokenize(text):
    return sent_tokenize(text)

def parseText(data, lower=True, tokenize=True, lemmatize=True, removeStopWords=True, removePunctuation=True, removeShortWords=True, removeNumeric=True):
    if lower:
        data = data.lower()
    if removePunctuation:
        for char in PUNCTUATION:
            if char == ".":
                continue   # we need the full stops for sentence separation
            data = data.replace(char, " ")
    if tokenize:
        data = wordpunct_tokenize(data)  # tokenize (also deals with \n issues)
    if removeNumeric:   # actually convert to whole literals
        data = [w if not w.isdigit() else inflectEngine.number_to_words(w)
                for w in data]
    if lemmatize:
        # lemmatize from plural to single
        data = [wordnet_lemmatizer.lemmatize(text) for text in data]
    if removeStopWords:
        data = [w for w in data if not w in stopwords.words('english')]
    if removeShortWords:
        data = [w for w in data if len(w) > 1]
    return data


def simpleCleanText(text):
    s = "".join([char if char in NORMALSYMBOLS+[" "]
                 else " " for char in list(text)])
    return s


def readTxtFile(file, encoding="UTF-8"):
    f = open(file, 'r', encoding=encoding)
    text = f.read()
    f.close()
    return text


def readJsonFile(file):
    with open(file) as data_file:
        data = json.load(data_file)
    return data


def filesInPath(mypath, ext=[]):
    if len(ext) == 0:
        onlyfiles = [f for f in listdir(mypath) if  f != ".DS_Store"]
    else:
        onlyfiles = [f for f in listdir(mypath) if f != ".DS_Store" and f[-len(ext):] == ext]
    return onlyfiles


def writeTxtFile(text, output):
    f = open(output, 'w')
    f.write(text)
    f.close
