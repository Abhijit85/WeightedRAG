"""In-memory Vector Index Tree built with FAISS."""

import faiss
import numpy as np
from data_classes import Data

"""
A node represents a dense embedding of a table, paragraph, or image

How do we represent these forms of data within our search structure?
    - Paragraph and tables have a common underlying modality: text. They'll be treated the same and
    the parsers will do the heavy lifting.

    - We can represent images with a caption (or metadata generation if needed). We'll want to store the path to the file somewhere.
"""


class IndexNode:
    def __init__(self, dim: int):
        """
        Initialize the Index Node

        Args:
            dim (int): Dimensionality of the vectors.
        """

        self.index = faiss.IndexFlatIP(dim)
        self.data: [Data | IndexNode] = []

    def write(self, embedding: np.array, data: Data | IndexNode):
        """
        Writes data to the store.

        Args:
            embedding (np.array): The embedding vector being added to the index.
            data (Data|StoreNode): The data associated with the embedding being added.
        """

        if embedding.ndim > 1:
            raise Exception("Invalid dimensions of embedding vector.")

        self.index.add(faiss.normalize_L2(embedding).reshape(1, -1))
        self.data.append(data)

    def retrieve(self, query_embedding: np.array, k: int = 10) -> [str]:
        """
        Performs a k-nearest neighbor retrieval.

        Args:
            query (np.array): The embedding of the query being used for retrieval.
            k (int): The amount of nearest neighbors to retrieve.
        """
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding)
        idx = 0
        distances, indices = (D[idx], I[idx])

        retrieved_data = []
        for dist, idx in zip(distances, indices):
            data = self.data[idx]
            retrieved_data.append(data)

        return retrieved_data

    def save(self):
        """
        Saves store to a file for future use.
        """
        raise NotImplementedError

    def __exit__(self):
        self.save()
