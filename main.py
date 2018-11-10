# 导入futuquant api
import futuquant as ft
import pandas as pd
import numpy as np


class StockDataLoader(object):

    def __init__(self):
        # 实例化行情上下文对象
        self.quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)  # type: ft.OpenQuoteContext
        # 上下文控制
        # self.quote_ctx.start()  # 开启异步数据接收
        # self.quote_ctx.set_handler(ft.TickerHandlerBase())  # 设置用于异步处理数据的回调对象(可派生支持自定义)

    def load_history(self):
        code = 'HK.00700'
        start = '2014-06-20'
        end = '2014-06-30'
        retdata = self.quote_ctx.get_history_kline(code, start=start, end=end, ktype=ft.KLType.K_DAY, autype=ft.AuType.QFQ)
        retdata = retdata[1]  # type: pd.DataFrame
        print(retdata.columns.values.tolist())
        for row in np.asarray(retdata).tolist():
            print(row)


def main():
    sdl = StockDataLoader()
    sdl.load_history()


if __name__ == '__main__':
    main()
