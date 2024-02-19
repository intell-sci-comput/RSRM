import numpy as np
from pandas import read_csv

from model.config import Config
from model.pipeline import Pipeline

def pipeline(csv_1, csv_2):
    """
    Expression generation function in demo, input csv address, output the best expression and error by RSRM.
    :param csv_1: train csv file split by comma
    :param csv_2: test csv file split by comma
    :param const: True for using parameters with parameter optimization, False vice versa
    """
    csv1, csv2 = read_csv(csv_1, header=None), read_csv(csv_2, header=None)
    x, t = np.array(csv1).T[:-1], np.array(csv1).T[-1]
    x_test, t_test = np.array(csv2).T[:-1], np.array(csv2).T[-1]
    config = Config()
    config.json("../config/config.json")
    config.set_input(x=x, t=t, x_=x_test, t_=t_test)
    model = Pipeline(config=config)
    best_exp, rmse = model.fit()
    print(f'\nresult: {best_exp} {rmse}')


if __name__ == '__main__':
    pipeline("../data/nguyen/11_train.csv", "../data/nguyen/11_test.csv")
