import random
import numpy as np

from model.config import Config


class DoubleQLearningNode:
    """
    double q-learning nodes
    """

    def __init__(self, actions):
        self.children = {}
        self.value_a = np.zeros(actions)
        self.value_b = np.zeros(actions)
        self.actions = actions

    def choose_max(self, dis):
        """
        :param dis: index of unused child
        :return: node with the highest value
        """
        value = self.value_b + self.value_a
        value[dis] = -1
        mx = value == np.max(value)
        action = np.random.choice(range(self.actions), p=mx / np.sum(mx))
        return action

    def choose_softmax(self, dis):
        """
        :param dis: index of unused child
        :return: sample of nodes by softmax value
        """
        value = self.value_b + self.value_a
        value[dis] = -1e9
        value -= np.max(value)
        soft = np.exp(value)
        action = np.random.choice(range(self.actions), p=soft / np.sum(soft))
        return action

    def softmax_possibility(self, dis):
        """
        :param dis: index of unused child
        :return: probabilities of nodes by softmax value
        """
        value = self.value_b + self.value_a
        value -= np.max(value)
        soft = np.exp(value)
        soft[dis] = 0
        soft /= np.sum(soft)
        return soft

    def check_exist(self, action):
        """
        check if action is chosen before , if not, generate new DoubleQLearningNode
        """
        if action not in self.children:
            self.children[action] = DoubleQLearningNode(self.actions)

    def learn(self, gamma, lr, a, r, finished):
        """
        Learning the node's reward
        :param gamma: decay rate
        :param lr: learning rate
        :param a: child node index
        :param r: reward
        :param finished: True if finished, else False
        """
        self.check_exist(a)
        if np.random.uniform() <= 0.5:
            q_predict = self.value_a[a]
            if not finished:
                q_target = r + gamma * np.max(self.children[a].value_b)  # next state is not terminal
            else:
                q_target = r  # next state is terminal
            self.value_a[a] += lr * (q_target - q_predict)  # update
        else:
            q_predict = self.value_b[a]
            if not finished:
                q_target = r + gamma * np.max(self.children[a].value_a)  # next state is not terminal
            else:
                q_target = r  # next state is terminal
            self.value_b[a] += lr * (q_target - q_predict)  # update


class DoubleQLearningTable:
    def __init__(self, actions, config_s: Config):
        self.actions = actions
        self.lr = config_s.mcts.q_learning_rate
        self.gamma = config_s.mcts.q_learning_discount
        self.epsilon = config_s.mcts.q_learning_epsilon
        self.root = DoubleQLearningNode(actions)
        self.q_table = self.root

    def choose_action(self, dis):
        """
        select action by choose_softmax within possibility of epsilon else random choose one
        :param dis: index of unused child
        :return:
        """
        # action selection
        if np.random.uniform() < self.epsilon:
            action = self.q_table.choose_softmax(dis)
        else:
            # choose random action
            st = set(range(self.actions)) - set(dis)
            action = random.sample(list(st), 1)[0]
        return action

    def learn(self, a, r, finished):
        """
        Learning the node's reward
        :param a: child node(action) index
        :param r: reward
        :param finished: True if finished, else False
        """
        self.q_table.learn(self.gamma, self.lr, a, r, finished)

    def get_status(self):
        """
        Get the status of the double Q-learning Table
        """
        return self.q_table

    def set_status(self, q):
        """
        Set the status of the double Q-learning Table
        """
        self.q_table = q

    def clear(self):
        """
        Clear double Q-learning Table
        """
        self.q_table = self.root

    def step(self, action):
        """
        step in an action and change the current double Q-learning node
        """
        self.q_table.check_exist(action)
        self.q_table = self.q_table.children[action]

    def possibility(self, dis):
        """
        :param dis: index of unused child
        :return: probabilities of nodes by softmax value
        """
        p = self.q_table.softmax_possibility(dis)
        return p
