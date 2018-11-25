import talib


def bbands(close=None, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """
    BOLL（林线指标）
    :param close: 收盘价
    :param timeperiod: 时间周期
    :param nbdevup: 上轨线标准差倍数
    :param nbdevdn: 下轨线标准差倍数
    :param matype: 移动平均类型
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: 上布林，中布林，下布林
    """
    upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return upper, middle, lower


def macd(close=None, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD（移动平均收敛/散度）
    指标解读：核心是看macdhist，柱子由正变负代表卖出信号，柱子由负变正代表买入信号
    :param close: 收盘价
    :param fastperiod: 短周期
    :param slowperiod: 长周期
    :param signalperiod: 移动平均周期
    :return: macd值，macd9日平均值，macd直方图高度
    """
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod,
                                            signalperiod=signalperiod)
    return macd, macdsignal, macdhist


def ma(close=None, timeperiod=5, matype=0):
    """
    MA（移动平均线）
    :param close: 收盘价
    :param timeperiod: 时间周期
    :param matype: 移动平均类型
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: MA平均线值
    """
    maval = talib.MA(close, timeperiod=timeperiod, matype=matype)
    return maval


def kd(high, low, close, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    """
    KDJ（随机指标）
    :param high: 最高价
    :param low: 最低价
    :param close: 收盘价
    :param fastk_period: 快速随机线K周期
    :param slowk_period: 慢速随机线K周期
    :param slowk_matype: 移动平均类型
    :param slowd_period: 慢速随机线d周期
    :param slowd_matype: 移动平均类型
    :return: 快速线K，慢速线D
    """
    k, d = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period,
                       slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)
    return k, d


def rsi(close, timeperiod=14):
    """
    RSI（相对强弱指标）
    :param close: 收盘价
    :param timeperiod: 移动平均周期
    :return: RSI值
    """
    rsival = talib.RSI(close, timeperiod=timeperiod)
    return rsival


def mavol(volume=None, timeperiod=5, matype=0):
    """
    MA（成交量移动平均线）
    :param volume: 成交量
    :param timeperiod: 时间周期
    :param matype: 移动平均类型
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA)
    :return: MAVOL平均线值
    """
    maval = talib.MA(volume, timeperiod=timeperiod, matype=matype)
    return maval


def ema(close=None, timeperiod=5):
    """
    MA（指数移动平均线）
    :param close: 收盘价
    :param timeperiod: 时间周期
    :return: EMA平均线值
    """
    maval = talib.EMA(close, timeperiod=timeperiod)
    return maval


def sar(high, low, acceleration=2, maximum=20):
    """
    SAR（停损转向操作点指标）
    (1)当股价上涨时,SAR的红色圆圈位于股价的下方,当该股的收盘价向下跌破SAR时,则应立即停损卖出。
    (2)当股价下跌时,SAR的绿色圆圈位于股价的上方,当收盘价向上突破SAR时,可以重新买回。
    :param high: 最高价
    :param low: 最低价
    :param acceleration: 加速因子步长
    :param maximum: 最大值
    :return: SAR值
    """
    sarval = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    return sarval


def ad(high, low, close, volume):
    """
    AD（多空双方力量浮标）
    :param high: 最高价
    :param low: 最低价
    :param close: 收盘价
    :param volume: 成交量
    :return: AD值
    """
    adval = talib.AD(high, low, close, volume)
    return adval


def adosc(high, low, close, volume, fastperiod=3, slowperiod=10):
    """
    ADOSC（收集派发摆荡指标）
    AD LINE的３天EMA值与10天EMA值的差值。这个指标是一个以０为中心的摆荡指标。这种计算手法与MACD的计算方法相似。
    :param high: 最高价
    :param low: 最低价
    :param close: 收盘价
    :param volume: 成交量
    :param fastperiod: 快线周期
    :param slowperiod: 慢线周期
    :return: ADOSC值
    """
    adoscval = talib.ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)
    return adoscval


def obv(close, volume):
    """
    OBV（能量潮指标）
    （1）当股价上升而OBV线下降，表示买盘无力，股价可能会回跌。
    （2）股价下降时而OBV线上升，表示买盘旺盛，逢低接手强股，股价可能会止跌回升。
    （3）OBV线缓慢上升，表示买气逐渐加强，为买进信号。
    （4）OBV线急速上升时，表示力量将用尽为卖出信号
    :param close:
    :param volume:
    :return:
    """
    obvval = talib.OBV(close, volume)
    return obvval
