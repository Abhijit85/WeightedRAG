"""In-memory Vector Index Tree built with FAISS."""

from threading import ExceptHookArgs
from .index_trie_node import IndexNode
import faiss
import numpy as np
from data_classes import Data
import heapq


class IndexTrie:
    def __init__(self, dim: int):
        """
        Initializes the Index Trie

        Args:
            dim (int): Dimensionality of the embedding space.
        """

        self.dim: int = dim
        self.root: IndexNode = IndexNode(self.dim)

    def write(self, path_embeddings, path_metadata):
        """
        TODO
        """
        u = self.root
        while path_metadata:
            data = path_metadata.pop(0)
            node_title = data["title"]
            node_data = data["data"]

            embedding, path_embeddings = path_embeddings[0], path_embeddings[1:]
            if node_title in u.adj_dict:
                u = u.adj_dict[node_title]
            else:
                new_node = IndexNode(self.dim, node_title, node_data)
                u.add_child(np.array([embedding]), node_title, new_node)
                u = new_node

    def nn_query(self, q_embeddings, k=10):
        max_heap = [(0.0, "", self.root, [])]
        return_list = []

        gamma = 0.9
        # TODO: Determine if scoring function is good for problem:
        # https://www.readcube.com/library/7ee9a5eb-fb66-4cc1-8a03-0ab285484c09:adcc1553-fcf8-4d32-bd00-61436fc33858.
        scoring_function = lambda a: (
            -(
                sum(
                    [
                        ((gamma ** (len(new_score_list) - 1 - idx)) * i)
                        for idx, i in enumerate(new_score_list)
                    ]
                )
                / sum(
                    [
                        (gamma ** (len(new_score_list) - 1 - idx))
                        for idx in range(len(new_score_list))
                    ]
                )
            )
        )
        while max_heap and len(return_list) < k:
            avg_ip, nl_path, u, ip_list = heapq.heappop(max_heap)
            if len(u.text_value) > 0:
                if len(nl_path):
                    nl_path += ", "
                nl_path += f"{u.text_value}"

            if len(u.adj) > 0:
                child_vals = u.search_children(q_embeddings)
                for dist, node in child_vals:
                    new_score_list = ip_list + [dist]
                    try:
                        heapq.heappush(
                            max_heap,
                            (
                                scoring_function(new_score_list),
                                nl_path,
                                node,
                                new_score_list,
                            ),
                        )
                    except Exception as e:
                        print("Heap: ", max_heap)
                        print(
                            "Bad Search: ",
                            (
                                scoring_function(new_score_list),
                                nl_path,
                                node,
                                new_score_list,
                            ),
                        )
                        raise e

            elif u.data != None:
                return_list.append((-avg_ip, {"path": nl_path, "data": u.data}))

        return [path_obj for (_, path_obj) in return_list]

    def save(self):
        """
        Saves index trie to a file for future use.
        """
        raise NotImplementedError

    def __exit__(self):
        try:
            self.save()
        except Exception as e:
            print(e)
