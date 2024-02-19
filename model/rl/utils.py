from typing import Tuple, List

from sympy import expand, sympify

from model.config import Config
from model.expr_utils.calculator import prune_poly_c
from model.expr_utils.exp_tree import PreTree
from model.rl.agent import Agent


def get_expression_and_reward(agent: Agent, tokens: Tuple[int], config_s: Config) -> Tuple[Tuple[int], float, str]:
    """
    Compute the full expression of the subsequence and the reward
    :param agent: Agent object to compute the reward of functions
    :param config_s: config file
    :param tokens: the subsequence to be calculated
    :return: subsequence, reward, expression of the sequence
    """
    exp = PreTree()
    for token in tokens:
        exp.add_exp(config_s.exp_dict[token])
    symbols = exp.get_exp()
    if symbols.count('C'):
        symbols = prune_poly_c(symbols)
    return tokens, agent.reward(tree=exp, reward=False), str(expand(sympify(symbols)))


def copy_game(agent: Agent, action_set: List[int]) -> Agent:
    """
    copy expression trees
    :param agent: the expression tree to be copied
    :param action_set: sequence of tokens for the current expression tree
    :return: the same expression tree as the game parameter
    """
    ans = Agent(
        config_s=agent.config_s
    )
    ans.expressions = agent.get_exps_full()
    for token in action_set:
        ans.add_token(token)
    if agent.exp_last:
        ans.exp_last = agent.exp_last
    return ans
