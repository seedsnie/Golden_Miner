import fix_yahoo_finance as yf


def main():
    tickers = ['0700.HK', 'BABA', 'BIDU', 'GOOGL', 'FB', 'AMZN']
    start_date = '2018-01-01'
    end_date = '2018-09-10'
    data = yf.download(tickers=tickers, start=start_date, end=end_date)
    print(data.head())


if __name__ == '__main__':
    main()
