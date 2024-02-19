from json import load

import numpy as np

from model.expr_utils.utils import expression_dict


class Config:
    def __init__(self):
        self.symbol_tol_num = 0
        self.best_exp = None, 1e999

        self.x = None
        self.x_ = None
        self.t = None
        self.t_ = None
        self.has_const = None
        self.const_optimize = None
        self.exp_dict = None
        self.reward_end_threshold = None
        self.verbose = None
        self.num_of_var = None
        self.epoch = None
        self.tokens = ["Add", "Sub", "Mul", "Div", 'Exp', 'Log', 'Cos', 'Sin', 'Sqrt']

        class mcts:
            def __init__(self):
                self.reward_end_threshold = None
                self.q_learning_rate = None
                self.q_learning_discount = None
                self.q_learning_epsilon = None
                self.mcts_const = None
                self.max_const = None
                self.max_height = None
                self.max_token = None
                self.max_exp_num = None
                self.token_discount = None
                self.times = None
                self.n0 = None

        self.mcts = mcts()

        class gp:
            def __init__(self):
                self.tournsize = None
                self.max_height = None
                self.cxpb = None
                self.mutpb = None
                self.max_const = None
                self.pops = None
                self.times = None
                self.hof_size = None
                self.token_discount = None

        self.gp = gp()

        class msdb:
            def __init__(self):
                self.max_used_expr_num = None
                self.expr_ratio = None
                self.token_ratio = None
                self.form_type = None

        self.msdb = msdb()

    def set_input(self, *, x, t, x_, t_):
        self.x = np.array(x)
        self.x_ = np.array(x_)
        self.t = np.array(t)
        self.t_ = np.array(t_)
        self.num_of_var = x.shape[0]
        self.exp_dict = expression_dict(self.tokens, self.num_of_var, self.has_const)

    def config_base(self, *, epoch=100, has_const=True, const_optimize=True, tokens=None, verbose=False,
                    reward_end_threshold=1e-10):
        if not tokens:
            tokens = ["Add", "Sub", "Mul", "Div", 'Exp', 'Log', 'Cos', 'Sin']
        self.epoch = epoch
        self.const_optimize = const_optimize
        self.has_const = has_const
        self.tokens = tokens
        self.verbose = verbose
        self.reward_end_threshold = reward_end_threshold

    def config_mcts(self, *, max_const=8, q_learning_rate=1e-3, mcts_const=2 ** 0.5,
                    max_height=5, max_token=20, max_expr_num=250, token_discount=0.99, times=100,
                    q_learning_discount=0.95, q_learning_epsilon=0.6, mcts_min_visits=10):
        self.mcts.token_discount = token_discount
        self.mcts.mcts_const = mcts_const
        self.mcts.q_learning_rate = q_learning_rate
        self.mcts.max_const = max_const
        self.mcts.max_height = max_height
        self.mcts.max_token = max_token
        self.mcts.max_exp_num = max_expr_num
        self.mcts.times = times
        self.mcts.n0 = mcts_min_visits
        self.mcts.q_learning_discount = q_learning_discount
        self.mcts.q_learning_epsilon = q_learning_epsilon

    def config_gp(self, *, max_const=5, pops=500, times=30, tournsize=10, max_height=10, cxpb=0.1, mutpb=0.5,
                  hof_size=20, token_discount=0.99):
        self.gp.max_height = max_height
        self.gp.tournsize = tournsize
        self.gp.cxpb = cxpb
        self.gp.mutpb = mutpb
        self.gp.max_const = max_const
        self.gp.pops = pops
        self.gp.times = times
        self.gp.hof_size = hof_size
        self.gp.token_discount = token_discount

    def config_msdb(self, *, max_expr_num=10, expr_ratio=0.1, token_ratio=0.5, form_type=None):
        self.msdb.max_used_expr_num = max_expr_num
        self.msdb.expr_ratio = expr_ratio
        self.msdb.token_ratio = token_ratio
        self.msdb.form_type = form_type
        if not form_type:
            self.msdb.form_type = ['Add', "Mul", "Pow"]

    def init(self):
        self.config_msdb()
        self.config_mcts()
        self.config_gp()
        self.config_base()

    def json(self, filepath):
        with open(filepath, 'r') as f:
            js = load(f)
            self.config_base(**js['base'])
            self.config_mcts(**js['mcts'])
            self.config_gp(**js['gp'])
            self.config_msdb(**js['msdb'])
