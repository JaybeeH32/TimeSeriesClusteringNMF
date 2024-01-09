import random
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np 

class Node2Vec():

    def __init__(self, G, n_dim, n, length, p, q):

        #Graph G we want the embedding
        self.G = G

        #Parameters of the Skipgram model 
        self.n_dim = n_dim

        #Parameters of the Random Walk 
        self.n = n #number of walks per node
        self.length = length #maximum length of each walk
        self.p = p #parameter to influence the probability of revisiting the previous node
        self.q = q #parameter to influence the probability of visiting the neighbors of the previous node


    def BiasedRandomWalk(self):
        walks = []
        G, n, length, p, q = self.G, self.n, self.length, self.p, self.q

        for node in G.nodes():
            for _ in range(n):
                walk = [node]
                previous_node = None
                current_node = node 

                for _ in range(length - 1):
                    neighbors = list(G.neighbors(current_node))
                    
                    if len(neighbors) > 0:
                        if previous_node is None:
                            probabilities = [G[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
                            total = sum(probabilities)
                            probabilities = [prob/total for prob in probabilities]
                            next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                        else:
                            probabilities = []
                            for neighbor in neighbors:
                                weight = G[current_node][neighbor].get('weight', 1)
                                if neighbor == previous_node:
                                    probabilities.append(weight / p)
                                elif G.has_edge(previous_node, neighbor):
                                    probabilities.append(weight)
                                else:
                                    probabilities.append(weight / q)
                            
                            # Normalize the probabilities
                            total = sum(probabilities)
                            probabilities = [prob/total for prob in probabilities]

                            next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    else:
                        break  # no more neighbors to explore

                    walk.append(next_node)
                    previous_node, current_node = current_node, next_node

                walks.append(walk)

        #to not have a structure that could be learned by a network 
        random.shuffle(walks)        
        return walks


    def Skipgram(self):

        G, n_dim, n, length, p, q = self.G, self.n_dim, self.n, self.length, self.p, self.q

        walks = self.BiasedRandomWalk()
        str_walks = [[str(n) for n in walk] for walk in walks]

        model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
        model.build_vocab(str_walks)
        model.train(str_walks, total_examples=model.corpus_count, epochs=5)
        return model 
        
    def Skipgram_embeddings(self):

        model = self.Skipgram()
        embeddings = np.zeros((len(list(self.G.nodes())), self.n_dim))
        for i, node in enumerate(self.G.nodes()):
            embeddings[i,:] = model.wv[str(node)]
        return embeddings