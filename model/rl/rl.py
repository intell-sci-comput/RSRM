from copy import deepcopy

from model.rl.double_q import DoubleQLearningTable
from model.rl.agent import Agent
from model.rl.mcts import SearchTree
import model.rl.utils as utils


class RLPipeline:
    def __init__(self, config_s):
        self.search_tree = None
        self.config_s = config_s
        self.exp_dict = config_s.exp_dict
        self.agent = Agent(config_s=config_s)
        self.ql_table = DoubleQLearningTable(len(self.exp_dict), config_s=self.config_s)
        self.times = config_s.mcts.times

    def clear(self):
        """
        Empty the current expression list and double q-learning table
        """
        self.agent.expressions = []
        self.ql_table = DoubleQLearningTable(len(self.exp_dict), config_s=self.config_s)

    def learn(self, pop, p=1):
        """
        Learning already generated expressions
        :param pop: list of learned expression and its reward
        :param p: scaling
        """
        self.agent.reset()
        for i in pop: self.agent.step(i)
        reward = self.agent.reward() * p
        s = []
        for action in pop:
            self.ql_table.learn(action, reward, True)
            self.ql_table.step(action)
            s.append(action)

    def get_one(self, act):
        """
        Generate a single expression
        :param act: former action chosen by mcts
        """
        l = deepcopy(act)
        dis = self.agent.unavailable()
        while True:
            if self.search_tree.empty():
                self.search_tree.expand([i for i in range(len(self.exp_dict)) if i not in dis])
            a = self.search_tree.choose_zero()
            if a == -1:
                a = self.ql_table.choose_action(dis)
            l.append(a)
            self.search_tree.step(a)
            r, done, dis = self.agent.step(a)  # take action
            self.ql_table.learn(a, r, self.agent.tree.is_full())
            if done:
                break
        self.search_tree.update(self.agent.reward())

    def run(self):
        """
        Clear MCTS and generate times of expressions.
        """
        self.search_tree = SearchTree(self.config_s)
        self.ql_table.clear()
        action_set = []
        while not utils.copy_game(self.agent, action_set).tree.is_full():
            for _ in range(self.times):
                self.agent = utils.copy_game(self.agent, action_set)
                s1, s2 = self.ql_table.get_status(), self.search_tree.get_status()
                self.get_one(action_set)
                self.ql_table.set_status(s1)
                self.search_tree.set_status(s2)
            a = self.search_tree.choose()
            action_set.append(a)
            self.ql_table.step(a)
            self.search_tree.step(a)

    def get_expressions(self):
        """
        Get the best expressions of agent
        """
        return self.agent.get_exps()
