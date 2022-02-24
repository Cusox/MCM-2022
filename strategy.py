import random

import numpy as np
import math
import csv
import datetime
import matplotlib.pyplot as plt
import pandas as pd


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

gold_flag = -1
bitcoin_flag = -1
gold_keep_days = 0
bitcoin_keep_days = 0

money = [1000, 0, 0]


# 计算前num天
def date_cal(now_time, day_num):
    return now_time + datetime.timedelta(days=day_num)


def data_return(start_time, end_time, series):
    data = []
    if series == 'gold':
        while start_time <= end_time:
            if gold_price_dict[start_time] is not None:
                data.append(gold_price_dict[start_time])
            start_time = date_cal(start_time, 1)
    elif series == 'bitcoin':
        while start_time <= end_time:
            data.append(bitcoin_price_dict[start_time])
            start_time = date_cal(start_time, 1)
    return data


def aver_line(start_time, end_time, series):
    data = data_return(start_time, end_time, series)
    return sum(data) / len(data)


def max_line(start_time, end_time, series):
    data = data_return(start_time, end_time, series)
    return max(data)


def min_line(start_time, end_time, series):
    data = data_return(start_time, end_time, series)
    return min(data)


def buy(flex_money, series):
    print(money)
    if series == 'gold':
        money[1] += flex_money / df['gold_close'].values[-1] / 1.01
    elif series == 'bitcoin':
        money[2] += flex_money / df['bitcoin_close'].values[-1] / 1.02
    money[0] -= flex_money
    return


def sell(series):
    print(money)
    if series == 'gold':
        if money[1] != 0:
            money[0] += money[1] * df['gold_close'].values[-1] * 0.99
            money[1] = 0
    elif series == 'bitcoin':
        if money[2] != 0:
            money[0] += money[2] * df['bitcoin_close'].values[-1] * 0.98
            money[2] = 0
    return


# 计算hurst值
def hurst_rs(data_raw):
    data = []
    hurst = []
    for r in range(1, len(data_raw)):
        data.append(np.log(data_raw[r] / data_raw[r - 1]))
    for n in range(2, len(data) // 2 + 1):
        A = len(data) // n
        x_a_i = [data[i: i + n] for i in range(0, len(data), n)]
        x_a = [sum([x_a_i[a][i] for i in range(n)]) / n for a in range(A)]
        s_a = [math.sqrt(sum([(x_a_i[a][i] - x_a[a]) ** 2 for i in range(n)]) / n) for a in range(A)]
        x_a_ii = [[sum((x_a_i[a][j] - x_a[a]) for j in range(i)) for i in range(n)] for a in range(A)]
        r_a = [max(x_a_ii[a]) - min(x_a_ii[a]) for a in range(A)]
        r_s_n = sum((r_a[a] / s_a[a]) for a in range(A)) / A
        hurst.append(r_s_n)
    return np.polyfit(np.log(np.arange(2, len(data) // 2 + 1)), np.log(np.array(hurst)), 1)[0]


# 改为趋势型
def change_trend(now_time, flex_money, series):
    if series == 'gold':
        global gold_flag
        gold_short_yesterday = df['gold_ma10'].values[-2]
        gold_short_today = df['gold_ma10'].values[-1]
        gold_long_yesterday = df['gold_ma20'].values[-2]
        gold_long_today = df['gold_ma20'].values[-1]
        if gold_flag != 0:
            sell(series)
        if gold_long_yesterday < gold_short_yesterday and gold_long_today > gold_short_today:
            gold_flag = -1
            sell(series)
        if gold_short_yesterday < gold_long_yesterday and gold_short_today > gold_long_today:
            gold_flag = 0
            buy(flex_money, series)
    elif series == 'bitcoin':
        global bitcoin_flag
        bitcoin_short_yesterday = df['bitcoin_ma10'].values[-2]
        bitcoin_short_today = df['bitcoin_ma10'].values[-1]
        bitcoin_long_yesterday = df['bitcoin_ma30'].values[-2]
        bitcoin_long_today = df['bitcoin_ma30'].values[-1]
        if bitcoin_flag != 0:
            sell(series)
        if bitcoin_long_yesterday < bitcoin_short_yesterday and bitcoin_long_today > bitcoin_short_today:
            bitcoin_flag = -1
            sell(series)
        if bitcoin_short_yesterday < bitcoin_long_yesterday and bitcoin_short_today > bitcoin_long_today:
            bitcoin_flag = 0
            buy(flex_money, series)
    return


# 改为震荡型
def change_shake(now_time, flex_money, series):
    if series == 'gold':
        global gold_flag
        global gold_keep_days
        gold_today_price = df['gold_close'].values[-1]
        gold_middle_line = df['gold_ma100'].values[-1]
        gold_std = df['gold_close'].std()
        gold_low_line = gold_middle_line - gold_std
        gold_stop_price = aver_line(date_cal(now_time, -20 + gold_keep_days), date_cal(now_time, -1), 'gold')
        # gold_data = data_return(date_cal(now_time, -20), date_cal(now_time, -1), 'gold')
        # gold_data_mean = math.sqrt(
        #     sum((gold_data[i] - gold_middle_line) ** 2 for i in range(len(gold_data))) / len(gold_data))
        # 布林线
        if gold_flag != 1:
            sell(series)
        if gold_today_price < gold_low_line:
            gold_flag = 1
            buy(flex_money, series)
        if gold_keep_days <= 10:
            gold_keep_days += 1
        if gold_today_price > gold_stop_price:
            gold_flag = -1
            gold_keep_days = 0
            sell(series)
    elif series == 'bitcoin':
        global bitcoin_flag
        global bitcoin_keep_days
        bitcoin_today_price = df['bitcoin_close'].values[-1]
        bitcoin_middle_line = df['bitcoin_ma100'].values[-1]
        bitcoin_std = df['bitcoin_close'].std()
        bitcoin_low_line = bitcoin_middle_line - bitcoin_std
        bitcoin_stop_price = aver_line(date_cal(now_time, -20 + bitcoin_keep_days), date_cal(now_time, -1), series)
        # bitcoin_data = data_return(date_cal(now_time, -20), date_cal(now_time, -1), 'bitcoin')
        # bitcoin_data_mean = math.sqrt(
        #     sum((bitcoin_data[i] - bitcoin_middle_line) ** 2 for i in range(len(bitcoin_data))) / len(bitcoin_data))
        # 布林线
        if bitcoin_flag != 1:
            sell(series)
        if bitcoin_keep_days <= 10:
            bitcoin_keep_days += 1
        if bitcoin_today_price < bitcoin_low_line:
            bitcoin_flag = 1
            buy(flex_money, series)
        if bitcoin_today_price > bitcoin_stop_price:
            bitcoin_flag = -1
            gold_keep_days = 0
            sell(series)
    return


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
        end_price = start_price + start_price * (
                    mean * delta_t + sigma * epsilon * np.sqrt(delta_t))  # 根据几何布朗运动公式计算期末价格
        end_price = max(0, end_price)  # 价格应大于0
        simulated_price.append(end_price)  # 将算的的结果存入列表
    return simulated_price


if __name__ == '__main__':
    begin = datetime.date(2016, 9, 11)  # start
    end = datetime.date(2021, 9, 10)  # end
    begin_time = begin
    begin_calculate = date_cal(begin_time, 100)
    num = 0
    cal_price()
    data_x = []
    data_y = []
    while begin_time <= end:
        gold_price_dict[begin_time] = goldPriceRaw[num]
        bitcoin_price_dict[begin_time] = BitcoinPrice[num]
        date_list.append(begin_time)
        num += 1
        today_price = bitcoin_price_dict[begin_time]
        df = pd.DataFrame({
            'gold_close': gold_price_dict.values(),
            'bitcoin_close': bitcoin_price_dict.values()},
            index=date_list)
        df['gold_return'] = df['gold_close'].pct_change()
        df['bitcoin_return'] = df['bitcoin_close'].pct_change()
        df['gold_close'] = df['gold_close'].fillna(method='pad')
        gold_price_mean = df['gold_close'].mean()
        if begin_calculate <= begin_time:
            # print(begin_time)
            df['gold_ma10'] = df['gold_close'].rolling(10).mean()
            df['gold_ma20'] = df['gold_close'].rolling(20).mean()
            df['gold_ma100'] = df['gold_close'].rolling(100).mean()
            df['bitcoin_ma10'] = df['bitcoin_close'].rolling(10).mean()
            df['bitcoin_ma30'] = df['bitcoin_close'].rolling(30).mean()
            df['bitcoin_ma100'] = df['bitcoin_close'].rolling(100).mean()
            bitcoin_hurst = hurst_rs(data_return(date_cal(begin_time, -30), date_cal(begin_time, -1), 'bitcoin'))
            bitcoin_money = money[0]  # 当前所有资金
            # 判断是否是休市日
            if not (math.isnan(df['gold_close'].values[-1])):
                gold_hurst = hurst_rs(data_return(date_cal(begin_time, -30), date_cal(begin_time, -1), 'gold'))
                gold_return_mean = df['gold_return'].mean()
                gold_return_std = df['gold_return'].std()
                bitcoin_return_mean = df['bitcoin_return'].mean()
                bitcoin_return_std = df['bitcoin_return'].std()
                manager = 1 / 6.25  # 通过var模型得出
                gold_money = manager * money[0]  # 按比重分配可支配资金
                bitcoin_money = (1 - manager) * money[0]  # 按比重分配可支配资金
                if gold_hurst > 0.5:
                    change_trend(begin_time, gold_money, 'gold')
                elif gold_hurst < 0.5:
                    change_shake(begin_time, gold_money, 'gold')
            if bitcoin_hurst > 0.5:
                change_trend(begin_time, bitcoin_money, 'bitcoin')
            elif bitcoin_hurst < 0.5:
                change_shake(begin_time, bitcoin_money, 'bitcoin')
            step_time = begin_time
            while gold_price_dict[step_time] is None:
                step_time = date_cal(step_time, -1)
            data_y.append(money[0] + money[1] * gold_price_mean + money[2] * df['bitcoin_close'].values[-1])
        else:
            data_y.append(money[0])
            print("就你小子想赚钱是吧")
        # print(begin_time)
        # print(data_y[len(data_y) - 1])
        data_x.append(begin_time)
        begin_time = date_cal(begin_time, 1)
    plt.plot(data_x, data_y)
    plt.show()
    print(money[0] + money[1] * gold_price_dict[end] + money[2] * gold_price_dict[end])
