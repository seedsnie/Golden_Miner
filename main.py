import futuquant as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import technicals as techs
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


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
        print('load quotes count=', self.dfquotes.shape[0])

    def get_quotes(self):
        return self.dfquotes

    def get_quote_params(self):
        return self.quote_params


class StockFeatureExtract(object):

    def __init__(self, dfquotes, quote_params):
        self.dfquotes = dfquotes   # type: pd.DataFrame
        self.quote_params = quote_params  # type: dict
        self.feature_cols = ['label']

    def extract(self):
        """
        根据股票行情数据提取特征数据
        """
        self.extract_basics()
        self.extract_indexs()
        self.extract_labels()

    def get_features(self):
        print(self.feature_cols)
        dffeatures = self.dfquotes[self.feature_cols]
        dffeatures = dffeatures.dropna()
        return dffeatures, self.feature_cols[1:]

    def extract_basics(self):
        """
        提取股票的基础信息（市盈率，换手率等）
        """
        self.feature_cols.append('pe_ratio')
        self.feature_cols.append('turnover_rate')
        self.feature_cols.append('change_rate')

    def extract_indexs(self):
        """
        提取K线的一些常用量化指标特征（KDJ，MACD，RSI等）
        """
        dfquotes = self.dfquotes
        dfhigh = dfquotes['high']
        dflow = dfquotes['low']
        dfclose = dfquotes['close']
        dfvolume = dfquotes['volume']

        # BOLL
        dfboll_upper, dfboll_middle, dfboll_lower = techs.bbands(dfclose)
        dfboll = (dfclose - dfboll_middle) / (dfboll_upper - dfboll_middle)
        dfquotes['feature_boll'] = dfboll
        self.feature_cols.append('feature_boll')

        # MACD
        dfquotes['feature_macd_val'], dfquotes['feature_macd_signal'], dfquotes['feature_macd_hist'] = \
            techs.macd(dfclose)
        self.feature_cols.append('feature_macd_hist')
        self.feature_cols.append('feature_macd_val')
        self.feature_cols.append('feature_macd_signal')

        # SAR
        dfsar = techs.sar(dfquotes.high, dfquotes.low)
        dfsar = (dfsar - dfclose) / dfclose
        dfquotes['feature_sar'] = dfsar
        self.feature_cols.append('feature_sar')

        # 均线指标
        # MA，EMA
        timeperiods = [5, 10, 20, 30, 60, 120]
        for timeperiod in timeperiods:
            dfma = techs.ma(dfclose, timeperiod=timeperiod)
            dfema = techs.ema(dfclose, timeperiod=timeperiod)
            dfquotes['feature_ma' + str(timeperiod)] = dfma
            dfquotes['feature_ema' + str(timeperiod)] = dfema
            self.feature_cols.append('feature_ma' + str(timeperiod))
            self.feature_cols.append('feature_ema' + str(timeperiod))

        # 随机指标
        # KD，RSI
        dfkd_k, dfkd_d = techs.kd(dfhigh, dflow, dfclose)
        dfquotes['feature_kd_k'] = dfkd_k
        dfquotes['feature_kd_d'] = dfkd_d
        self.feature_cols.append('feature_kd_k')
        self.feature_cols.append('feature_kd_d')
        timeperiods = [5, 10, 20, 30, 60, 120]
        for timeperiod in timeperiods:
            dfrsi = techs.rsi(dfclose, timeperiod=timeperiod)
            dfquotes['feature_rsi' + str(timeperiod)] = dfrsi
            self.feature_cols.append('feature_rsi' + str(timeperiod))

        # 成交量指标
        # AD，ADOSC，OBV
        dfad = techs.ad(dfhigh, dflow, dfclose, dfvolume)
        dfadosc = techs.adosc(dfhigh, dflow, dfclose, dfvolume)
        dfobv = techs.obv(dfclose, dfvolume)
        dfquotes['feature_ad'] = dfad
        dfquotes['feature_adosc'] = dfadosc
        dfquotes['feature_obv'] = dfobv
        self.feature_cols.append('feature_ad')
        self.feature_cols.append('feature_adosc')
        self.feature_cols.append('feature_obv')

        # 显示指标图表
        # show_index = ['time_index', 'close', 'feature_sar']
        # self.show_index_table(show_index)

    def extract_labels(self):
        dfquotes = self.dfquotes
        dflabelcontexts = dfquotes['close'].to_frame()
        dflabelcontexts['close_next'] = dflabelcontexts.shift(-1)

        def apply_labels(close, close_next):
            label = 1 if close < close_next else 0
            return label
        dfquotes['label'] = dflabelcontexts.apply(lambda row: apply_labels(row[0], row[1]), axis=1)
        self.dfquotes = dfquotes[:-1]

    def show_index_table(self, show_index):
        dfquotes = self.dfquotes
        x_datetime = pd.to_datetime(dfquotes.time_key)
        dfquotes['time_index'] = x_datetime
        dfshow = dfquotes[show_index]
        ax = dfshow.plot(x='time_index', figsize=(20, 10), grid=True, title=self.quote_params['code'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 設置x軸主刻度顯示格式（日期）
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 設置x軸主刻度間距
        plt.show()


class MLFeatureProcess(object):

    def __init__(self, dffeatures, feature_names):
        self.dffeatures = dffeatures   # type: pd.DataFrame
        self.feature_names = feature_names

    def process(self):
        """
        将股票特征数据转化成符合机器学习规范的训练集和测试集
        :return: 数据集
        """
        dffeatures, dflabels = self.feature_ensemble()

        # 数据集划分
        counts = dffeatures.shape[0]
        train_split = 0.95
        train_index = int(round(train_split * counts))
        print('MLFeatureProcess samples total=%d train=%d test=%d' % (counts, train_index, (counts - train_index)))
        train_x, test_x = dffeatures[:train_index].values, dffeatures[train_index:].values
        train_y, test_y = dflabels[:train_index].values, dflabels[train_index:].values

        # 数据归一化处理
        train_x = preprocessing.scale(train_x)
        test_x = preprocessing.scale(test_x)

        dataset = {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y,
                   'feature_names': self.feature_names}

        return dataset

    def feature_ensemble(self):
        """
        特征融合（把多天特征融合到一个向量）
        """
        dffeatures = self.dffeatures[self.feature_names]
        dflabel = self.dffeatures['label']
        ensemble_days = 5
        dffeatures_ensembles = [dffeatures]
        for day in range(1, ensemble_days + 1):
            dfconcat = dffeatures.shift(day)
            feature_names_ago = [ name + '_ago' + str(day) for name in self.feature_names ]
            dfconcat.columns = feature_names_ago
            dffeatures_ensembles.append(dfconcat)
        dffeatures_ensembles = pd.concat(dffeatures_ensembles, axis=1)
        dffeatures_ensembles = dffeatures_ensembles[ensemble_days:]
        dflabels = dflabel[ensemble_days:]
        return dffeatures_ensembles, dflabels


class MLModel(object):

    def __init__(self, dataset):
        self.dataset = dataset   # type: dict
        self.feature_selection = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=50)
        self.feature_selection_switch = False
        # 模型配置
        self.classifier_name = 'GradientBoostingClassifier'
        if self.classifier_name == 'LogisticRegression':
            self.classifier = LogisticRegression(verbose=1)
        elif self.classifier_name == 'GradientBoostingClassifier':
            # learning_rate=0.05, max_depth=3, min_samples_leaf=90, n_estimators=40, subsample=0.5
            self.classifier = GradientBoostingClassifier(verbose=1, learning_rate=0.05, subsample=0.5, n_estimators=40)
        elif self.classifier_name == 'SVC':
            self.classifier = SVC()

    def train(self):
        # 加载数据
        dataset = self.dataset
        train_x, train_y = dataset['train_x'], dataset['train_y']

        # 特征筛选处理
        train_x_fs = train_x
        if self.feature_selection_switch:
            train_x_fs = self.feature_selection.fit_transform(train_x, train_y)
            select_features = self.feature_selection.get_support()  # type: np.ndarray
            print(list(zip(self.dataset['feature_names'], select_features.tolist())))

        # 模型训练
        self.classifier.fit(X=train_x_fs, y=train_y)
        # self.show_model()

    def search_model(self):
        # 加载数据
        dataset = self.dataset
        train_x, train_y = dataset['train_x'], dataset['train_y']
        param_sets = {}
        if self.classifier_name == 'GradientBoostingClassifier':
            param_sets = {'n_estimators': range(20, 81, 10), 'subsample': np.arange(0.5, 1.0, 0.1),
                          'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200),
                          'min_samples_leaf': range(60, 101, 10), 'learning_rate': [0.05, 0.1, 0.01]}
        search = GridSearchCV(estimator=self.classifier, param_grid=param_sets, n_jobs=8)
        search.fit(X=train_x, y=train_y)
        print(search.best_params_, search.best_score_)

    def metrics(self):
        dataset = self.dataset
        test_x, test_y = dataset['test_x'], dataset['test_y']
        test_x_fs = test_x
        if self.feature_selection_switch:
            test_x_fs = self.feature_selection.transform(test_x)
        pred_y = self.classifier.predict(X=test_x_fs)
        print(classification_report(y_true=test_y, y_pred=pred_y, target_names=['class_green', 'class_red']))

    def show_model(self):
        if self.classifier_name == 'LogisticRegression':
            print('\nshow model feature coef=', self.classifier.coef_)
        elif self.classifier_name == 'GradientBoostingClassifier':
            print('\nshow model feature importances=', self.classifier.feature_importances_)


def main():
    # 输入参数
    input_code = 'HK.00700'
    input_start = '2000-02-08'
    input_end = '2018-11-24'

    # 加载历史数据
    sdl = StockDataLoader(input_code, input_start, input_end)
    sdl.load_history()

    # 提取股票特征
    sfe = StockFeatureExtract(sdl.get_quotes(), sdl.get_quote_params())
    sfe.extract()

    # 模型训练和效果评估
    dffeatures, feature_names = sfe.get_features()
    mlfp = MLFeatureProcess(dffeatures, feature_names)
    dataset = mlfp.process()
    model = MLModel(dataset)
    model.train()
    model.metrics()


if __name__ == '__main__':
    main()
