import numpy as np
import math
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def lagrange(x, y, num_points, x_test):
    # 所有的基函数值，每个元素代表一个基函数的值
    l = np.zeros(shape=(num_points,))

    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值
        # 由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            # 这里没搞清楚，书中公式上没有对k=k_时，即分母为0进行说明
            # 有些资料上显示k是不等于k_的
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k] * (x_test - x[k_]) / (x[k] - x[k_])
            else:
                pass
                # 计算当前需要预测的x_test对应的y_test值
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i] * l[i]
    return L


workbookGold = csv.reader(open('LBMA-GOLD.csv', 'r'))
workbookBitcoin = csv.reader(open('BCHAIN-MKPRU.csv', 'r'))
goldPrice = np.array([])
BitcoinPrice = np.array([])
date = []


def cal_price():
    global goldPrice
    global BitcoinPrice
    global date
    # bitcoin price
    for i in workbookBitcoin:
        date.append(i[0])
        BitcoinPrice = np.append(BitcoinPrice, i[1])
    date = date[1:]
    BitcoinPrice = BitcoinPrice[1:]
    BitcoinPrice = list(map(float, BitcoinPrice))
    BitcoinPriceSave = BitcoinPrice

    # gold price,use lagrange interpolation,10 nearest points to fill the missing data
    iterator = 0
    for i in workbookGold:
        if i[0] == 'Date':
            continue
        elif i[0] == date[iterator] and i[1] != '':
            goldPrice = np.append(goldPrice, i[1])
            iterator += 1
        elif i[0] == date[iterator] and i[1] == '':
            goldPrice = np.append(goldPrice, '0')
            iterator += 1
        elif i[0] != date[iterator] and i[1] == '':
            goldPrice = np.append(goldPrice, '0')
            iterator += 1
        else:
            while i[0] != date[iterator]:
                goldPrice = np.append(goldPrice, '-1')
                # print("i:",i[0])
                # print("date:",date[iterator])
                iterator += 1
            goldPrice = np.append(goldPrice, i[1])
            iterator += 1
    goldPriceEnum = list(enumerate(goldPrice))
    # print(goldPriceEnum)
    blank = list(filter(lambda x: x[1] == '0', goldPriceEnum))
    print("missing postion in list 'goldPrice':", blank)
    amount = 4  # the amount of nearest points to fill the missing data
    for i in range(0, len(blank), 2):
        leftarr = []
        countl = 0
        posl = 0
        while countl != amount:
            if goldPrice[blank[i][0] - countl - posl - 1] != '-1':
                leftarr.append(goldPrice[blank[i][0] - countl - posl - 1])
                countl += 1
            else:
                posl += 1
        leftarr = leftarr[::-1]
        rightarr = []
        countr = 0
        posr = 0
        while countr != amount:
            if goldPrice[blank[i + 1][0] + countr + posr + 1] != '-1':
                rightarr.append(goldPrice[blank[i + 1][0] + countr + posr + 1])
                countr += 1
            else:
                posr += 1
        count = 0
        while goldPrice[blank[i][0] + 1 + count] == '-1':
            count += 1
        goldPrice[blank[i][0]] = lagrange(list(np.delete(np.linspace(1, 12, 12), [5, 8])),
                                          list(
                                              map(lambda x: float(x), leftarr + [goldPrice[blank[i][0] + count + 1]] + [
                                                  goldPrice[blank[i + 1][0] - 1]] + rightarr)),
                                          10,
                                          6)
        goldPrice[blank[i + 1][0]] = lagrange(list(np.delete(np.linspace(1, 12, 12), [5, 8])),
                                              list(map(lambda x: float(x),
                                                       leftarr + [goldPrice[blank[i][0] + count + 1]] + [
                                                           goldPrice[blank[i + 1][0] - 1]] + rightarr)),
                                              10,
                                              9)
    goldPrice = goldPrice.tolist()
    for i in range(len(goldPrice)):
        goldPrice[i] = float(goldPrice[i])
    for i in range(len(goldPrice)):
        if goldPrice[i] == -1:
            goldPriceRaw.append(None)
        else:
            goldPriceRaw.append(goldPrice[i])


date_list = []
goldPriceRaw = []
gold_price_dict = {}
bitcoin_price_dict = {}
weights = np.array([.37, .63])


# 计算前num天
def date_cal(now_time, day_num):
    return now_time + datetime.timedelta(days=day_num)


# 返回一段日期内的价格序列
def data_return(start_time, end_time):
    data = []
    while start_time <= end_time:
        data.append(bitcoin_price_dict[start_time])
        start_time = date_cal(start_time, 1)
    return data


def GBM(s_price, mean, sigma, T, n):
    """计算几何布朗运动的价格数据

       parameters:
           s_0: 开始价格
           mu: 观察期日收益率的均值
           sigma: 观察期日收益率的标准差
           T: 预测价格的周期长度,如预测下一天，T=1，预测后10天，T=10；
           n: 单次模拟的步数，步数越大，模拟得越精确；

   """

    delta_t = T / n  # 计算delta_t
    simulated_price = [s_price]  # 创建一个空的列表用于储存价格数据
    # 模拟价格走势
    for i in range(n):
        start_price = simulated_price[i]  # 获取期初价格
        epsilon = np.random.normal()  # 按照标准正态分布产生一个随机数
        end_price = start_price + start_price * (mean * delta_t + sigma * epsilon * np.sqrt(delta_t))  # 根据几何布朗运动公式计算期末价格
        end_price = max(0, end_price)  # 价格应大于0
        simulated_price.append(end_price)  # 将算的的结果存入列表
    return simulated_price


if __name__ == '__main__':
    begin = datetime.date(2016, 9, 11)  # start
    end = datetime.date(2021, 9, 10)  # end
    begin_time = begin
    begin_gbm = date_cal(begin_time, 30)
    num = 0
    cal_price()
    while begin_time <= end:
        gold_price_dict[begin_time] = goldPriceRaw[num]
        bitcoin_price_dict[begin_time] = BitcoinPrice[num]
        date_list.append(begin_time)
        num += 1
        begin_time = date_cal(begin_time, 1)
    gold_today_price = gold_price_dict[end]
    bitcoin_today_price = bitcoin_price_dict[end]
    df = pd.DataFrame({
        'gold_close': gold_price_dict.values(),
        'bitcoin_close': bitcoin_price_dict.values()},
        index=date_list)
    df['gold_close'] = df['gold_close'].fillna(method='pad')
    df['gold_return'] = df['gold_close'].pct_change()
    df['bitcoin_return'] = df['bitcoin_close'].pct_change()
    gold_return_mean = df['gold_return'].mean()
    gold_return_std = df['gold_return'].std()
    bitcoin_return_mean = df['bitcoin_return'].mean()
    bitcoin_return_std = df['bitcoin_return'].std()
    simulated_prices_gold = []
    for i in range(10000):
        # 模拟一次几何布朗运动
        simulated_price = GBM(gold_today_price, gold_return_mean, gold_return_std, 1, 100)
        # 取出最终价格
        final_price = simulated_price[-1]
        simulated_prices_gold.append(final_price)
    simulated_return_gold = [simulated_prices_gold[i] / gold_today_price - 1 for i in range(len(simulated_prices_gold))]
    gold_VaR_1 = np.percentile(simulated_return_gold, 1)
    gold_VaR_10 = gold_VaR_1 * np.sqrt(10)  # 平方根法则，用一天的VaR估算十天的VaR
    simulated_prices_bitcoin = []
    for i in range(10000):
        # 模拟一次几何布朗运动
        simulated_price = GBM(bitcoin_today_price, bitcoin_return_mean, bitcoin_return_std, 1, 100)
        # 取出最终价格
        final_price = simulated_price[-1]
        simulated_prices_bitcoin.append(final_price)
    simulated_return_bitcoin = [simulated_prices_bitcoin[i] / bitcoin_today_price - 1 for i in
                                range(len(simulated_prices_bitcoin))]
    bitcoin_VaR_1 = np.percentile(simulated_return_bitcoin, 1)
    bitcoin_VaR_10 = bitcoin_VaR_1 * np.sqrt(10)  # 平方根法则，用一天的VaR估算十天的VaR
    gold_lost = gold_VaR_1 * gold_today_price
    bitcoin_lost = bitcoin_VaR_1 * bitcoin_today_price
    print(bitcoin_VaR_1 / gold_VaR_1) # 范围在 5~5.5之间，取中为5.25，得出比特币和黄金的可支配资金的比例为1：5.25
    # plt.figure(figsize=(8, 6))
    # df.gold_return.hist(bins=50, alpha=0.6, color='red')
    # df.bitcoin_return.hist(bins=50, alpha=0.6, color='steelblue')
    # plt.axvline(gold_VaR_10, color='r', linewidth=1, label='gold_VaR_10')
    # plt.axvline(gold_VaR_1, color='y', linewidth=1, label='gold_VaR_1')
    # plt.axvline(bitcoin_VaR_10, color='b', linewidth=1, label='bitcoin_VaR_10')
    # plt.axvline(bitcoin_VaR_1, color='g', linewidth=1, label='bitcoin_VaR_1')
    # plt.legend()
    # plt.xlabel('return')
    # plt.ylabel('count')
    # plt.show()
