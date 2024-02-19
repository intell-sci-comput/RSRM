from collections import Counter
from time import time

from model.expr_utils.utils import FinishException
from model.ga.ga import GAPipeline
from model.rl.rl import RLPipeline
from model.rl.utils import get_expression_and_reward
from model.msdb.msdb import MSDB
from model.config import Config


class Pipeline:
    def __init__(self, config: Config = None):
        self.config: Config = config
        if config is None:
            self.config = Config()
            self.config.init()
        self.expr_form = ""
        self.ga1, self.ga2 = None, None
        self.rl1, self.rl2 = None, None
        self.msdb = None

    def fit(self, x=None, t=None, x_test=None, t_test=None):
        if self.config.x is None:
            self.config.set_input(x=x, t=t, x_=x_test, t_=t_test)
        self.ga1, self.ga2 = GAPipeline(self.config), GAPipeline(self.config)
        self.rl1, self.rl2 = RLPipeline(self.config), RLPipeline(self.config)
        self.msdb = MSDB(config_s=self.config)
        try:
            sym_tol1 = []
            sym_tol2 = []
            tm_start = time()
            for tms in range(self.config.epoch):
                if self.config.verbose:
                    print(
                        f'\rEpisode: {tms + 1}/{self.config.epoch}, time: {round(time() - tm_start, 2)}s, '
                        f'expression: {self.config.best_exp[0]}, loss: {self.config.best_exp[1]}, '
                        f'form: {self.expr_form}F', end='')
                if tms % 25 == 0:
                    self.rl1.clear()
                    sym_tol1.clear()
                if tms % 30 == 0:
                    self.rl2.clear()
                    sym_tol2.clear()
                if tms % 10 <= 8:
                    self.rl1.run()
                    pop = self.ga1.ga_play(self.rl1.get_expressions())
                    sym_tol1 += pop
                    self.change_expr_form(pop)
                if tms % 10 >= 5:
                    self.rl2.run()
                    pop = self.rl2.get_expressions()
                    sym_tol2 += pop
                    pop = self.ga2.ga_play(pop)
                    sym_tol2 += pop
                if tms % 10 >= 7:
                    sym_tol2 += self.ga2.ga_play(sym_tol2)
                if tms % 10 == 5:
                    self.ga1.ga_play(sym_tol1)
        except FinishException:
            pass
        return self.config.best_exp

    def change_expr_form(self, pops):
        pops = [tuple(p) for p in pops]
        symbols = [get_expression_and_reward(self.rl2.agent, pop, self.config) for pop in set(pops)]
        symbols_count = Counter(pops)
        self.expr_form = self.msdb.get_form(symbols, symbols_count)
        if self.expr_form:
            self.rl2.agent.change_form(self.expr_form)
            self.ga2.agent.change_form(self.expr_form)
