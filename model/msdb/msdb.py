from collections import defaultdict
from typing import List, Counter, Set, Tuple

from model.config import Config


def process_symbol(symbol: str) -> Set[str]:
    """
    Split a symbol expression into sub-expressions using '+' '-'
    :param symbol: expression
    :return: collection of subexpressions
    """
    symbols = []
    now = ""
    quote = 0
    for s in symbol:
        if s in '({[':
            quote += 1
        if s in ')}]':
            quote -= 1
        if s in '+-' and 0 == quote:
            symbols.append(now)
            now = ""
        if s not in ' ':
            now += s
    symbols.append(now)
    return set(map(lambda x: x[1:] if x.startswith('+') else x, symbols))


class MSDB:
    """
    an instance of the modulated subtree discovery block
    """

    def __init__(self, config_s: Config):
        self.config = config_s

    def __symbol_add(self, symbols: List[Tuple[Tuple[int], float, str]], symbols_count: Counter[Tuple[int]]):
        """
        Compute the add expression form , i.e., find A in T = A + f(x)
        :param symbols: the best expressions and rewards
        :param symbols_count: the count of symbols
        :return: subtree form as "A+F(x)" where A = A(X)
        """
        symbols.sort(key=lambda x: x[1])
        symbols = symbols[:self.config.msdb.max_used_expr_num]
        st = symbols[0][1]
        sym_dict = defaultdict(int)
        tm = 0
        if abs(st) < 1e-99:
            return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            if st / val > self.config.msdb.expr_ratio or idx <= 1:
                for s in process_symbol(symbol):
                    if not s:
                        continue
                    sym_dict[s] += symbols_count[tokens]
                tm += 1
        syms_now = ""
        for symbol, times in sym_dict.items():
            if times >= tm * self.config.msdb.token_ratio:
                if not syms_now or symbol.startswith('-'):
                    syms_now += symbol
                else:
                    syms_now += '+' + symbol
        if syms_now:
            syms_now = syms_now + '+'
        return syms_now

    def __symbol_mul(self, symbols: List[Tuple[Tuple[int], float, str]]):
        """
        Compute the multiply expression form , i.e., find A in T = A * f(x)
        :param symbols: the best expressions and rewards=
        :return: subtree form as "A*F(x)" where A = A(X)
        """
        symbols.sort(key=lambda x: x[1])
        syms_now = ""
        st = symbols[0][1]
        if abs(st) < 1e-99:
            return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            flag = 1
            for sym in process_symbol(symbol):
                if (not sym.startswith('C*') and not sym == 'C') or len(sym) >= 3:
                    flag = 0
            if flag:
                syms_now = f"({symbol})"
                break
        if syms_now:
            syms_now = syms_now + '*'
        return syms_now

    def __symbol_pow(self, symbols: List[Tuple[Tuple[int], float, str]]):
        """
        Compute the power expression form , i.e., find A in T = A ** f(x)
        :param symbols: the best expressions and rewards=
        :return: subtree form as "A**F(x)" where A = A(X)
        """
        symbols.sort(key=lambda x: x[1])
        syms_now = ""
        st = symbols[0][1]
        if abs(st) < 1e-99:
            return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            flag = 1
            for sym in process_symbol(symbol):
                if not sym.count('**') or len(sym) >= 3:
                    flag = 0
            if flag:
                syms_now = f"({symbol.split('**')[0]})"
                break
        if syms_now:
            syms_now = syms_now + '**'
        return syms_now

    def get_form(self, symbols: List[Tuple[Tuple[int], float, str]], symbols_count: Counter[Tuple[int]]):
        """
        Compute the add/mul/pow expression form
        :param symbols: the best expressions and rewards
        :param symbols_count: the count of symbols
        :return: subtree form
        """
        if ('Mul' in self.config.msdb.form_type) and (form := self.__symbol_mul(symbols)):
            return form
        if ('Pow' in self.config.msdb.form_type) and (form := self.__symbol_pow(symbols)):
            return form
        if ('Add' in self.config.msdb.form_type) and (form := self.__symbol_add(symbols, symbols_count)):
            return form
        return ""
