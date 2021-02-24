import pandas as pd
import numpy as np
import random


def calculate_monthly_profit(data_frame):
    if type(data_frame) == list:
        return [data_frame[i + 1] / data_frame[i] for i in range(len(data_frame) - 1)]
    else:
        return [data_frame.Valor[i + 1] / data_frame.Valor[i] for i in range(len(data_frame.Valor) - 1)]


def calculate_acc_profit(monthly_profit):
    acc_profit = monthly_profit.copy()
    acc_profit.iloc[0] = monthly_profit.iloc[0]
    for i in range(1, len(monthly_profit)):
        acc_profit.iloc[i] = monthly_profit.iloc[i]*acc_profit.iloc[i-1]

    return acc_profit


def sample_monthly_profit(monthly_profit, num_samples):
    start = random.randrange(0, len(monthly_profit) - num_samples + 1)
    stop = start + num_samples - 1
    sampled_monthly_profit = monthly_profit.iloc[start:stop+1]
    return sampled_monthly_profit


def import_data():
    cdi_data = pd.read_csv("../Dados/CDI.csv")
    ifix_data = pd.read_csv("../Dados/IFIX.csv")
    ibov_data = pd.read_csv("../Dados/IBOVESPA.csv")
    sp500_usd_data = pd.read_csv("../Dados/SP500_dolar.csv")
    usd_data = pd.read_csv("../Dados/USD_BRL.csv")
    sp500_data = [a*b for a, b in zip(sp500_usd_data.Valor, usd_data.Valor)]
    # sp500_data = sp500_usd_data  # Without convert (wrong approach, just for visualization)

    cdi_monthly_profit = (cdi_data.Valor/100 + 1)**(1/12)
    ifix_monthly_profit = calculate_monthly_profit(ifix_data)
    ibov_monthly_profit = calculate_monthly_profit(ibov_data)
    sp500_monthly_profit = calculate_monthly_profit(sp500_data)

    monthly_profit = pd.DataFrame({'CDI': cdi_monthly_profit, 'IFIX': ifix_monthly_profit,
                                   'IBOVESPA': ibov_monthly_profit, 'SP500': sp500_monthly_profit})

    return monthly_profit


class DataManager:
    def __init__(self):
        self.monthly_profit = import_data()
        self.acc_profit = calculate_acc_profit(self.monthly_profit)

    def calculate_ark_profit(self, fracs, additional_args=None):
        """
        :param fracs: [frac_cdi, frac_ifix, frac_ibov]
        :type fracs: Numpy array.
        :param additional_args: [do_mont_carlo, num_samples, samples_size]
        :type additional_args: List.

        """
        if additional_args is None or not additional_args[0]:
            additional_args = [False, 1, len(self.monthly_profit)]

        num_samples = additional_args[1]
        samples_size = additional_args[2]
        if sum(fracs) > 1:  # frac_cdi + frac_ifix + frac_ibov + frac_sp500 = 1 ->
                            # -> frac_cdi + frac_ifix + frac_ibov < 1
            return -1  # Gives a negative "reward"

        fracs = np.append(fracs, [1 - sum(fracs)])

        ark_acc_profit = []
        for _ in range(num_samples):
            sampled_monthly_profit = sample_monthly_profit(self.monthly_profit, samples_size)
            ark_acc_profit.append(sampled_monthly_profit.dot(fracs).product())
        return sum(ark_acc_profit)/len(ark_acc_profit)
