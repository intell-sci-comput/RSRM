import copy
import random
from typing import Dict, List, Tuple

from model.config import Config
from model.expr_utils.exp_tree import LevelTree
from model.expr_utils.calculator import cal_expression
from model.expr_utils.exp_tree_node import Expression


class Agent:
    """
    Monte Carlo Tree Search Algorithm Implementation Class
    """

    def __init__(self, config_s: Config):
        self.config_s = config_s

        self.tree = LevelTree()
        self.expressions: List[Tuple[float, List[int]]] = []
        self.exp_last: str = ""

        self.max_parameter = config_s.mcts.max_const
        self.max_height = config_s.mcts.max_height
        self.max_token = config_s.mcts.max_token
        self.discount = config_s.mcts.token_discount
        self.expression_dict: Dict[int, Expression] = config_s.exp_dict

    def get_exps_full(self, num=None) -> List[Tuple[float, List[int]]]:
        """
        Get the best num expression and loss value.
        """
        if num is None:
            num = self.config_s.mcts.max_exp_num
        ans = sorted(self.expressions, key=lambda x: -x[0])[:num]
        self.expressions = []
        return ans

    def get_exps(self, num=None) -> List[List[int]]:
        """
        Get the best num expression.
        """
        if num is None:
            num = self.config_s.mcts.max_exp_num
        ans = sorted(self.expressions, key=lambda x: -x[0])[:num]
        tol = [i[1] for i in ans]
        self.expressions = []
        return tol

    def reward(self, tree=None, reward=True):
        """
        Compute the current expression in the expression tree reward
        :param tree: input expression tree
        :param reward: return discount ** length_of_expr / (1 + rmse) if reward is True else rmse
        :return: reward or rmse of tree
        """
        try:
            if not tree:
                tree = self.tree
            if not tree.is_full():
                print(tree.token_list_pre)
                raise RuntimeError
            pre = tree.token_list_pre
            length_of_expr = len(pre)
            if length_of_expr <= 5:
                return 0 if reward else 1e999
            symbols = tree.get_exp()
            if self.exp_last:
                symbols = f"{self.exp_last}({symbols})"
            ans_tol = cal_expression(symbols, self.config_s)
            val = self.discount ** length_of_expr / (1 + ans_tol)
            if pre not in [i[1] for i in self.expressions]:
                self.expressions.append((ans_tol, pre))
        except TimeoutError:
            val = 0
            ans_tol = 1e999
        if reward:
            return val
        return ans_tol

    def reset(self):
        """
        reset expression tree
        """
        self.tree = LevelTree()
        return [0], self.unavailable()

    def unavailable(self) -> List[int]:
        """
        Calculate the set of currently unselectable tokens
        """
        self.tree.trim()
        exps = []
        ans = []
        if self.tree.depth() > self.max_height:
            return [i for i, j in self.expression_dict.items() if j.child != 0]
        if self.tree.tri_count > 0 or self.tree.depth() <= 0:
            exps.extend(["Cos", "Sin"])
        if self.tree.head_token == "Exp":
            exps.append('Log')
        if self.tree.head_token == "Log":
            exps.append('Exp')
        if self.tree.const_num == self.max_parameter:
            exps.append("C")
        for i, j in self.expression_dict.items():
            if i not in ans and j.type_name in exps:
                ans.append(i)
        return ans

    def predict(self) -> float:
        """
        Randomly fill the whole expression tree to compute the expectation reward
        """
        tol = range(len(self.expression_dict))
        tree = copy.deepcopy(self.tree)
        tree, self.tree = self.tree, tree
        while not self.tree.is_full():
            action = random.choice(list(set(tol) - set(self.unavailable())))
            self.add_token(action)
        reward = self.reward()
        self.tree = tree
        return reward

    def change_form(self, expr) -> None:
        """
        change the current expression form
        """
        self.exp_last = expr

    def add_token(self, token) -> None:
        """
        add new token to the expression tree
        """
        self.tree.add_exp(self.expression_dict[token])

    def step(self, token) -> Tuple[float, bool, List[int]]:
        """
        Add new token
        :return: reward, end or not, indexes of unselectable child nodes
        """
        self.add_token(token)
        if self.tree.is_full():
            return self.reward(), True, []
        unavail = self.unavailable()
        return - 0.1, False, unavail
