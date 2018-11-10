import talib


def bbands(closes=None, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    布林线指标计算
    :param closes: 收盘价
    :param timeperiod: 时间周期
    :param nbdevup: 上轨线标准差倍数
    :param nbdevdn: 下轨线标准差倍数
    :param matype: 移动平均类型
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return:
    """
    upper, middle, lower = talib.BBANDS(closes, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return upper, middle, lower
