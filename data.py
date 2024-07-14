import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import os
from yahoo_fin import stock_info as si
import time


def pull_options_data(tickers: list[str], read_from_data: bool=False, date=None):
    if read_from_data:
        return pd.read_csv(f"data_folder/{date}_options_data.csv")
    
    
        
    def calculate_rolling_vol(ticker, window=21):
        try:
            start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") 
            end = datetime.today().strftime("%Y-%m-%d") 
            historical_data = yf.download(ticker, start=start, end=end)
            returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
            rolling_volatility = returns.rolling(window=window).std().iloc[-1]
            rolling_annualized_volatility = rolling_volatility * np.sqrt(252)
            return rolling_annualized_volatility
        except:
            return np.nan

    total_data = pd.DataFrame()


    for i, t in enumerate(tickers):
        print(f"({i + 1} / {len(tickers)}) Downloading for {t}...")
        try:
            ticker = yf.Ticker(t)
        
            option_expiration_dates = ticker.options
        except:
            continue

        ticker_data = pd.DataFrame()

        for expiration_date in option_expiration_dates:

            options_data = ticker.option_chain(expiration_date)
            
            options_data.calls['putOrCall'] = "call"
            options_data.puts['putOrCall'] = "put"

            options = pd.concat([options_data.calls, options_data.puts])
        
            if options.empty:
                continue
            start_date = options['lastTradeDate'].min().strftime('%Y-%m-%d')
            end_date = options['lastTradeDate'].max().strftime('%Y-%m-%d')
            historical_data = ticker.history(start=start_date, end=end_date)
            if len(historical_data) == 0:
                continue

            historical_data.index = pd.to_datetime(historical_data.index).date

            options['lastTradeDate'] = pd.to_datetime(options['lastTradeDate']).dt.date

            def get_underlying_price(trade_date):
                if trade_date in historical_data.index:
                    return historical_data.loc[trade_date]['Close']
                else:
                    nearest_date = min(historical_data.index, key=lambda d: abs(d - trade_date))
                    return historical_data.loc[nearest_date]['Close']

            options['underlyingPrice'] = options['lastTradeDate'].apply(get_underlying_price)

            options['expiryDate'] = pd.to_datetime(expiration_date).date()
            options['timeToMaturity'] = (options['expiryDate'] - options['lastTradeDate']).apply(lambda x: x.days)

            options['moneyness'] = options['underlyingPrice'] / options['strike']

            options['optionType'] = options['putOrCall'].apply(lambda x: 1 if x == "call" else 0)

            ticker_data = pd.concat([ticker_data, options])

        if not ticker_data.empty:
            historical_vol = calculate_rolling_vol(t)
            if not np.isnan(historical_vol):
                ticker_data['historicalVolatility'] = historical_vol
                total_data = pd.concat([total_data, ticker_data])
        
    treasury_data = yf.Ticker("^TNX").history(period="5d")
    risk_free_rate = treasury_data['Close'].iloc[-1] / 100 
    total_data['riskFreeRate'] = risk_free_rate
    
    total_data.to_csv(f"data_folder/{datetime.today().strftime('%Y%m%d')}_options_data.csv")
    return total_data

def get_all_options_data():
    total_data = pd.DataFrame()
    for file in os.listdir("data_folder"):
        data = pd.read_csv(os.path.join("data_folder", file))
        total_data = pd.concat([total_data, data])
    return total_data.reset_index(drop=True)


class OptionsDataset(Dataset):

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


    


if __name__ == '__main__':
    sp500_tickers = si.tickers_sp500()
    nasdaq_tickers = si.tickers_nasdaq()
    dow_tickers = si.tickers_dow()
    other_tickers = si.tickers_other()


    all_tickers = list(set(sp500_tickers + nasdaq_tickers + dow_tickers + other_tickers))
    tickers = [s for s in all_tickers if s.strip()]

    pull_options_data(tickers)
