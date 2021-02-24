import pandas as pd
import numpy as np


def calculate_monthly_profit(data_frame):
    if type(data_frame) == list:
        return [data_frame[i + 1] / data_frame[i] for i in range(len(data_frame) - 1)]
    else:
        return [data_frame.Valor[i + 1] / data_frame.Valor[i] for i in range(len(data_frame.Valor) - 1)]


def import_data():
    cdi_data = pd.read_csv("../Dados/CDI.csv")
    ifix_data = pd.read_csv("../Dados/IFIX.csv")
    ibov_data = pd.read_csv("../Dados/IBOVESPA.csv")
    sp500_usd_data = pd.read_csv("../Dados/SP500_dolar.csv")
    usd_data = pd.read_csv("../Dados/USD_BRL.csv")
    sp500_data = [a/b for a, b in zip(sp500_usd_data.Close, usd_data.Close)]

    cdi_monthly_profit = cdi_data.Valor/100 + 1
    ifix_monthly_profit = calculate_monthly_profit(ifix_data)
    ibov_monthly_profit = calculate_monthly_profit(ibov_data)
    sp500_monthly_profit = calculate_monthly_profit(sp500_data)

    monthly_profit = pd.DataFrame({'CDI': cdi_monthly_profit, 'IFIX': ifix_monthly_profit,
                                   'IBOVESPA': ibov_monthly_profit, 'SP500': sp500_monthly_profit})

    return monthly_profit


class DataManager:
    def __init__(self):
        self.monthly_profit = import_data()

    def calculate_total_profit(self, fracs):
        """
        :param fracs: [frac_cdi, frac_ifix, frac_ibov]
        :type fracs: Numpy array.
        """
        if sum(fracs) > 1:
            return 0

        fracs = np.append(fracs, [1 - sum(fracs)])

        acc_profit = self.monthly_profit.dot(fracs).product()
        return acc_profit
