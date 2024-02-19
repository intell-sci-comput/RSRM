import random
from typing import Sequence

from model.config import Config


class TreeNode:
    """
    MCTS nodes
    """

    def __init__(self, father):
        self.father = father
        self.sons = None
        self.max_v = -1e9
        self.times = 0

    def update(self, v: float) -> None:
        """
        Recursively update the max and mean value from bottom to top
        :param v: value
        """
        now = self
        while now:
            now.max_v = max(now.max_v, v)
            now = now.father

    def expand(self, sons: Sequence[int]) -> None:
        """
        Expand sons of a node
        :param sons: index of sons to expand
        """
        self.sons = {
            son: TreeNode(self) for son in sons
        }

    def choose_max(self):
        """
        Choose the nodes with max value
        :return: its index
        """
        return max(self.sons.items(), key=lambda x: x[1].max_v)[0]

    def choose_zero(self, n0):
        """
        Choose the nodes with visits less than n0
        :return: its index
        """
        st = [i[0] for i in self.sons.items() if i[1].times < n0]
        if not st:
            return -1
        return random.choice(st)


class SearchTree:
    def __init__(self, config: Config):
        self.root = TreeNode(None)
        self.now = self.root
        self.n0 = config.mcts.n0

    def empty(self):
        """
        check if the tree is empty(to expand)
        """
        return self.now.sons is None

    def expand(self, sons):
        """
        Expand sons of a node
        """
        self.now.expand(sons)

    def choose(self):
        """
        Choose the nodes with max value
        """
        return self.now.choose_max()

    def update(self, v):
        """
        Recursively update the max and mean value from bottom to top
        :param v: value
        """
        self.now.update(v)

    def clear(self):
        """
        return to the top of the tree
        """
        self.now = self.root

    def step(self, a):
        """
        change current node to its son with index a
        :param a: child index
        """
        self.now = self.now.sons[a]
        self.now.times += 1

    def get_status(self):
        """
        Get the status of the MCTS
        """
        return self.now

    def set_status(self, now):
        """
        Set the status of the MCTS
        """
        self.now = now

    def choose_zero(self):
        """
        Choose the nodes with visits less than n0
        :return: its index
        """
        return self.now.choose_zero(self.n0)
