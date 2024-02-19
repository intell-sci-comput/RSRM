from typing import Dict, Tuple

from model.config import Config
from model.expr_utils.exp_tree import PreTree
from model.expr_utils.calculator import cal_expression
import model.ga.utils as utils
from model.expr_utils.exp_tree_node import Expression


class Agent:
    """
    Genetic Algorithm Implementation Class
    """

    def __init__(self, toolbox, config_s):
        self.config_s: Config = config_s
        self.toolbox = toolbox

        self.exp_last: str = ""

        self.max_parameter = config_s.gp.max_const
        self.discount = config_s.gp.token_discount
        self.expression_dict: Dict[int, Expression] = config_s.exp_dict

    def change_form(self, expr: str) -> None:
        """
        Replacing the new expression pattern
        :param expr: mode
        """
        self.exp_last = expr

    def fitness(self, individual) -> Tuple[float,]:
        """
        Calculate the fitness of the individual(expression)
        """
        try:
            tree = PreTree()
            token_list = utils.deap_to_tokens(individual)
            if len(token_list) <= 5:
                return 1e999,
            for token in token_list:
                if token not in self.available(tree):
                    return 1e999,
                tree.add_exp(self.expression_dict[token])
            symbols = tree.get_exp()
            if self.exp_last:
                symbols = f"{self.exp_last}({symbols})"
            ans = cal_expression(symbols, self.config_s)
            val = self.discount ** (-len(individual)) * ans  # 计算适应度 越小越好
            return val,
        except TimeoutError:
            pass
        return 1e999,

    def available(self, tree):
        """
        Determining the usable nodes of a tree
        :param tree: expression tree
        """
        exps = []
        ans = list(self.expression_dict.keys())
        if tree.tri_count > 0:
            exps.extend(["Cos", "Sin"])
        if tree.head_token == "Exp":
            exps.append('Log')
        if tree.head_token == "Log":
            exps.append('Exp')
        if tree.const_num == self.max_parameter:
            exps.append("C")
        for i, j in self.expression_dict.items():  # calculate excluded nodes
            if i in ans and j.type_name in exps:
                ans.remove(i)
        return ans
