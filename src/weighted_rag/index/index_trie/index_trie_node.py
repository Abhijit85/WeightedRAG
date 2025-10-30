"""In-memory Vector Index Tree Node built with FAISS."""

import faiss
import numpy as np
from .data_classes import Data

"""
A node represents a dense embedding of a table, paragraph, or image

How do we represent these forms of data within our search structure?
    - Paragraph and tables have a common underlying modality: text. They'll be treated the same and
    the parsers will do the heavy lifting.

    - We can represent images with a caption (or metadata generation if needed). We'll want to store the path to the file somewhere.
"""


class IndexNode:
    def __init__(self, dim: int, text_value="", data=None):
        """
        Initialize the Index Node

        Args:
            dim (int): Dimensionality of the vectors.
        """

        self.index = faiss.IndexFlatIP(dim)
        self.text_value = text_value
        self.data = data
        self.adj_dict: [Data | IndexNode] = {}
        self.adj: [Data | IndexNode] = []

    def add_child(self, embedding: np.array, node: IndexNode, data: Data | IndexNode):
        """
        Writes data to the store.

        Args:
            embedding (np.array): The embedding vector being added to the index.
            data (Data|StoreNode): The data associated with the embedding being added.
        """

        if embedding.ndim > 1:
            raise Exception("Invalid dimensions of embedding vector.")

        self.index.add(embedding)
        self.adj.append(node)
        self.adj_dict[data] = node

    def search_children(self, q_embeddings: np.array):
        """
        Searches children based on query embedding.

        Args:
            embedding (np.array): The embedding vector being added to the index.
            data (Data|StoreNode): The data associated with the embedding being added.
        """
        avg_q_embedding = np.mean(q_embeddings, axis=0, keepdims=True)
        D, I = self.index.search(faiss.normalize_L2(avg_q_embedding), len(self.adj))
        idx = 0
        distances, indices = (D[idx], I[idx])
        res = []
        for dist, idx in zip(distances, indices):
            node = self.adj[idx]
            res.append((dist, node))
        return res

    def __lt__(self, other):
        """Less than comparison, primarily used for heap operations"""
        if not isinstance(other, IndexNode):
            raise NotImplementedError
        return self.text_value < other.text_value

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, IndexNode):
            raise NotImplementedError
        return self.text_value == other.text_value and self.data == other.data

    def save(self):
        """
        Saves store to a file for future use.
        """
        raise NotImplementedError

    def __exit__(self):
        try:
            self.save()
        except Exception as e:
            print(e)
