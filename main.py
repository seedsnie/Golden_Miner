import futuquant as ft
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import matplotlib.dates as mdates
import mpl_finance as mpf
import datetime
import technicals as techs


class StockDataLoader(object):

    def __init__(self, code='', start='', end=''):
        # 实例化行情上下文对象
        self.quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)  # type: ft.OpenQuoteContext
        self.dfquotes = None  # type: pd.DataFrame
        self.quote_params = {'code': code, 'start': start, 'end': end}
        # 上下文控制
        # self.quote_ctx.start()  # 开启异步数据接收
        # self.quote_ctx.set_handler(ft.TickerHandlerBase())  # 设置用于异步处理数据的回调对象(可派生支持自定义)

    def load_history(self):
        code = self.quote_params['code']
        start = self.quote_params['start']
        end = self.quote_params['end']
        retdata = self.quote_ctx.get_history_kline(code, start=start, end=end, ktype=ft.KLType.K_DAY,
                                                   autype=ft.AuType.QFQ)
        self.dfquotes = retdata[1]  # type: pd.DataFrame

    def get_quotes(self):
        return self.dfquotes

    def get_quote_params(self):
        return self.quote_params


class StockFeatureExtract(object):

    def __init__(self, dfquotes, quote_params):
        self.dfquotes = dfquotes   # type: pd.DataFrame
        self.quote_params = quote_params # type: dict

    def extract(self):
        self.extract_indexs()

    def extract_indexs(self):
        """
        提取K线的一些常用量化指标特征（KDJ，MACD，RSI等）
        """
        dfquotes = self.dfquotes
        dfquotes['upper'], dfquotes['middle'], dfquotes['lower'] = techs.bbands(dfquotes.close.values, timeperiod=20,
                                                                                nbdevup=2, nbdevdn=2, matype=0)
        x_datetime = pd.to_datetime(dfquotes.time_key)
        dfquotes['time_index'] = x_datetime
        dfshow = dfquotes[['time_index', 'close', 'upper', 'middle', 'lower']]
        ax = dfshow.plot(x='time_index', figsize=(20, 10), grid=True, title=self.quote_params['code'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 設置x軸主刻度間距
        plt.show()


def main():
    # 输入参数
    input_code = 'HK.00700'
    input_start = '2018-01-01'
    input_end = '2018-11-09'

    # 加载历史数据
    sdl = StockDataLoader(input_code, input_start, input_end)
    sdl.load_history()

    # 提取股票特征
    sfe = StockFeatureExtract(sdl.get_quotes(), sdl.get_quote_params())
    sfe.extract()


if __name__ == '__main__':
    main()
