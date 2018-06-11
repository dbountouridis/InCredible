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
import string
import random
from deap import creator, base, tools, algorithms


# convert individual of population to a sentence grouping
def seriesSentence(sentences,individual):
	s=sentences[:]
	v=individual[:]
	ns=[]
	t=s[0]
	for i in range(len(v)):
		if v[i]==0:
			ns.append(t)
			t=s[i+1]
		else:
			t+=" "+s[i+1]
	ns.append(t)
	return ns

# evaluation function
def evalOneMax(individual):
	s=np.loadtxt('temp/vector.in')
	v=individual[:]
	ns=[]
	t=[s[0]]
	for i in range(len(v)):
		if v[i]==0:
			ns.append(t)
			t=[s[i+1]]
		else:
			t.append(s[i+1])
	ns.append(t)
	sum_=[np.sum(series) for series in ns]
	
	
	S=-np.std(sum_)-(100*(len(ns)<5)) # minimum 5 groups!!!!!!
	# print(individual,ns,sum_,np.std(sum_),S)
	# time.sleep(1)
	# #print(individual,S)
	return (float(S),)

# the main ga function
def GAonSequences(vector,population=50,generations=50):
    np.savetxt('temp/vector.in', np.array(vector), delimiter=',',fmt='%f',) 
    random.seed(2)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(vector)-1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=population)
    #print(population)
    NGEN=generations
    for gen in range(NGEN):
    	#print(" generation ",gen)
    	offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    	fits = toolbox.map(toolbox.evaluate, offspring)
    	for fit, ind in zip(fits, offspring):
    		#print fit,ind
    		ind.fitness.values = fit
    	population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)
    #print("Top 10 shifts...")
    # for top in top10:
    # 	print(top,evalOneMax(top))
    return top10[0]







    

    

