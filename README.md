# InCredible

A Python script for locating cross-referenced pieces of information in related documents (e.g. articles from different outlets of the same news story). The original news dataset from [Kaggle](https://www.kaggle.com/snapcrack/all-the-news).

Run a demo
```
python demo.py
```

The script performs the following:
1. Reads the documents, parses them into sentences.
2. Uses a naive GA (genetic algorithm) to group sentences into "meaningful" pieces of information (POIs).
3. Computes the claim similarity between POIs.
4. Creates an undirected graph from the POIs.
5. Compute the cliques in the graph, corresponding to cross-referenced pieces of information.

### Cite
If you use the code, please cite the following publication:

D. Bountouridis, M. Marrero, N. Tintarev, C. Hauff, _Explaining Credibility in News Articles using Cross-Referencing_, 
2018 Workshop on ExplainAble Recommendation and Search (EARS 2018)
