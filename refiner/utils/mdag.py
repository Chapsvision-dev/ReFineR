from typing import List, Tuple, Union, Dict
import numpy as np


class MDAG:
    def __init__(
        self, edges: List[Tuple[List[str]]], vertices: Union[List[str], None] = None
    ):
        self._vertices: Dict[str, List[List[int]]] = {}
        self._edges: List[Tuple[List[str]], List[str]] = []

        if vertices is None:
            vertices = []
            for e in edges:
                vertices += e[0] + e[1]
        vertices = list(set(vertices))

        self._add_vertices(vertices)
        self._add_edges(edges)
        self._compute_orders()

    @property
    def sources(self):
        return [
            v for v in self._vertices if len(self._vertices[v][1]) == 0
        ]  # A source is defined as a vertex not made from any vertex

    def _add_vertices(self, vertices: List[str]):
        """
        vertices: used to, made from (theoretically must have a single 'made from' edge)
        """
        for v in vertices:
            self._vertices[v] = [[], []]

    def _add_edges(self, edges: List[Tuple[List[str], List[str]]]):
        """
        edges: output, input
        """
        for e in edges:
            i = len(self._edges)
            self._edges.append(e)
            for v in e[0]:  # For each input add to 'made from'
                self._vertices[v][1].append(i)
            for v in e[1]:  # For each output add to 'used to'
                self._vertices[v][0].append(i)

    def _compute_orders(self):
        """
        Finds the order of each vertex and edge
            For a vertex it is defined as the number of steps to reach it from a source vertex
            For a (multi) edge it is defined as the max order of its outputs
        """
        seeds = self.sources
        orders = {v: 0 for v in self._vertices}

        def rec(seeds: List[int], orders: Dict[str, int]) -> Dict[str, int]:
            for s in seeds:
                for e in self._vertices[s][0]:  # For each edge 'used to'
                    outs = self._edges[e][0]  # Outputs from edge
                    for o in outs:
                        orders[o] = max(orders[o], orders[s] + 1)
                    orders = rec(outs, orders)
            return orders

        self._vertices_orders = rec(seeds, orders)
        self._edges_orders = [
            max([self._vertices_orders[v] for v in e[0]]) for e in self._edges
        ]

    @property
    def index_edges_orders(self) -> List[int]:
        r = list(range(len(self._edges)))
        return [r[i] for i in np.argsort(self._edges_orders).tolist()]

    def extract_from_output(self, output: List[str]):
        """
        Extract a computation graph from the desired output
        """

        def rec(
            seed: List[str],
            vertices: List[str],
            edges: List[Tuple[List[str], List[str]]],
            edge_indices: List[int],
        ) -> Tuple[List[str], List[Tuple[List[str], List[str]]], List[int]]:
            for v in seed:
                if v in vertices:  # If already computed
                    continue

                if v not in self._vertices and v not in vertices:
                    # If not in graph add floating vertex
                    vertices.append(v)
                    continue
                vertex = self._vertices[v]
                if len(vertex[1]) == 0:
                    # If source vertex
                    vertices.append(v)
                    continue

                # Find how to make the vertex using its 'made from' edge
                ei = self._vertices[v][1][0]
                edge = self._edges[ei]
                vertices += edge[0]
                edges.append(edge)
                edge_indices.append(ei)
                vertices, edges, edge_indices = rec(
                    edge[1], vertices, edges, edge_indices
                )
            return vertices, edges, edge_indices

        vertices, edges, edge_indices = rec(output, [], [], [])
        graph = MDAG(edges, vertices)
        return graph, edge_indices

    def prune_schedule_from_output(self, output: List[str]) -> Dict[str, int]:
        """
        Remove unused vertices after computation
        """
        m = len(self._edges)
        kills = {v: -1 for v in self._vertices}
        for i, ie in enumerate(reversed(self.index_edges_orders)):
            # For each edge / computation
            for v in self._edges[ie][1] + self._edges[ie][0]:
                # For each vertex 'made from' and each vertex 'used to'
                if v not in output:
                    kills[v] = max(kills[v], m - i - 1)
                else:
                    kills[v] = m
        for v in kills:
            if v in output and (v not in self._vertices or kills[v] == -1):
                kills[v] = m

        schedule = [[] for _ in range(m)]
        for v in kills:
            k = kills[v]
            if k < m:
                schedule[k].append(v)
        return schedule
