import talib


def bbands(closes=None, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    BOLL（林线指标）
    :param closes: 收盘价
    :param timeperiod: 时间周期
    :param nbdevup: 上轨线标准差倍数
    :param nbdevdn: 下轨线标准差倍数
    :param matype: 移动平均类型
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: 上布林，中布林，下布林
    """
    upper, middle, lower = talib.BBANDS(closes, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return upper, middle, lower


def macd(closes=None, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD（移动平均收敛/散度）
    指标解读：核心是看macdhist，柱子由正变负代表卖出信号，柱子由负变正代表买入信号
    :param closes: 收盘价
    :param fastperiod: 短周期
    :param slowperiod: 长周期
    :param signalperiod: 移动平均周期
    :return: macd值，macd9日平均值，macd直方图高度
    """
    macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=fastperiod, slowperiod=slowperiod,
                                            signalperiod=signalperiod)
    return macd, macdsignal, macdhist
