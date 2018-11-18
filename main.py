import futuquant as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import technicals as techs
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report


class StockDataLoader(object):

    def __init__(self, code='', start='', end=''):
        # 实例化行情上下文对象
        self.quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)  # type: ft.OpenQuoteContext
        self.dfquotes = None  # type: pd.DataFrame
        self.quote_params = {'code': code, 'start': start, 'end': end}

    def load_history(self):
        code = self.quote_params['code']
        start = self.quote_params['start']
        end = self.quote_params['end']
        retdata = self.quote_ctx.get_history_kline(code, start=start, end=end, ktype=ft.KLType.K_DAY,
                                                   autype=ft.AuType.QFQ)

        # code, time_key, open, close, high, low, pe_ratio ,turnover_rate,volume,turnover,change_rate,last_close
        self.dfquotes = retdata[1]  # type: pd.DataFrame

    def get_quotes(self):
        return self.dfquotes

    def get_quote_params(self):
        return self.quote_params


class StockFeatureExtract(object):

    def __init__(self, dfquotes, quote_params):
        self.dfquotes = dfquotes   # type: pd.DataFrame
        self.quote_params = quote_params  # type: dict

    def extract(self):
        """
        根据股票行情数据提取特征数据
        """
        self.extract_indexs()
        self.extract_labels()

    def get_features(self):
        cols = ['feature_boll_upper', 'feature_boll_middle', 'feature_boll_lower',
                'feature_macd_val', 'feature_macd_signal', 'feature_macd_hist',
                'label']
        dffeatures = self.dfquotes[cols]
        dffeatures = dffeatures.dropna()
        return dffeatures

    def extract_indexs(self):
        """
        提取K线的一些常用量化指标特征（KDJ，MACD，RSI等）
        """
        dfquotes = self.dfquotes

        # BOLL
        dfquotes['feature_boll_upper'], dfquotes['feature_boll_middle'], dfquotes['feature_boll_lower'] = \
            techs.bbands(dfquotes.close.values)
        # MACD
        dfquotes['feature_macd_val'], dfquotes['feature_macd_signal'], dfquotes['feature_macd_hist'] = \
            techs.macd(dfquotes.close.values)

        # self.show_index_table()

    def show_index_table(self):
        dfquotes = self.dfquotes
        x_datetime = pd.to_datetime(dfquotes.time_key)
        dfquotes['time_index'] = x_datetime
        show_index = ['time_index', 'close',
                      'feature_boll_upper', 'feature_boll_middle', 'feature_boll_lower',
                      'feature_macd_hist']
        dfshow = dfquotes[show_index]
        ax = dfshow.plot(x='time_index', figsize=(20, 10), grid=True, title=self.quote_params['code'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 設置x軸主刻度間距
        plt.show()

    def extract_labels(self):
        dfquotes = self.dfquotes
        dflabelcontexts = dfquotes['close'].to_frame()
        dflabelcontexts['close_next'] = dflabelcontexts.shift(1)

        def apply_labels(close, close_next):
            label = 1.0 if close < close_next else 0.0
            return label
        dfquotes['label'] = dflabelcontexts.apply(lambda row: apply_labels(row[0], row[1]), axis=1)
        self.dfquotes = dfquotes[:-1]


class MLFeatureProcess(object):

    def __init__(self, dffeatures):
        self.dffeatures = dffeatures   # type: pd.DataFrame

    def process(self):
        """
        将股票特征数据转化成符合机器学习规范的训练集和测试集
        :return: 数据集
        """
        dfsamples = self.dffeatures.sample(frac=1.0)   # type: pd.DataFrame
        dffeatures = dfsamples[['feature_boll_upper', 'feature_boll_middle', 'feature_boll_lower',
                                'feature_macd_val', 'feature_macd_signal', 'feature_macd_hist']]
        dflabel = dfsamples['label']

        counts = dfsamples.shape[0]
        train_split = 0.8
        train_index = int(round(train_split * counts))
        train_x, test_x = dffeatures[:train_index].as_matrix(), dffeatures[train_index:].as_matrix()
        train_y, test_y = dflabel[:train_index].as_matrix(), dflabel[train_index:].as_matrix()
        dataset = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y}

        return dataset


class MLModel(object):

    def __init__(self, dataset):
        self.dataset = dataset   # type: dict
        self.classifier = LogisticRegression()

    def train(self):
        dataset = self.dataset
        train_x, train_y = dataset['train_x'], dataset['train_y']
        self.classifier.fit(X=train_x, y=train_y)

    def metrics(self):
        dataset = self.dataset
        test_x, test_y = dataset['test_x'], dataset['test_y']
        pred_y = self.classifier.predict(X=test_x)
        print(classification_report(y_true=test_y, y_pred=pred_y, target_names=['class_red', 'class_green']))


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

    # 模型训练和效果评估
    mlfp = MLFeatureProcess(sfe.get_features())
    dataset = mlfp.process()
    model = MLModel(dataset)
    model.train()
    model.metrics()


if __name__ == '__main__':
    main()
