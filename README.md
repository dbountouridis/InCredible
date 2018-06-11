# NLPfunctions

Script for computing the trusted claims/sentences in a group of related news articles. The original news dataset from [Kaggle](https://www.kaggle.com/snapcrack/all-the-news).

Run a demo
```
python demo.py
```

The script performs the following:
1. Reads the documents, parses them into sentences.
2. Uses a naive GA (genetic algorithm) to group sentences into "meaningful" statements/claims.
3. Computes the claim similarity between documents.
4. Creates an undirected graph from the claims.
5. Compute the cliques in the graph.
6. Export the the graph with colored cliques in D3.js format.
7. Export the documents into and html format with claims colored by how much they are shared between documents.

Challenges:
1. Ensuring that the initial documents are diverse (not biased towards certain views).
2. Grouping sentences into meaningful claims.
3. Meaningful sentence similarity function.
4. Visualizing the information in an intuitive way.