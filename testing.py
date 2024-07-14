import torch
import QuantLib as ql
import joblib
import numpy as np
from pricing_model import PricingModel
from data import get_all_options_data
from sklearn.metrics import mean_squared_error

feature_columns = ['strike', 'historicalVolatility', 'timeToMaturity', 'moneyness', 'optionType', 'riskFreeRate']

scaler = joblib.load("scaler.pkl")
pricing_model = PricingModel(input_dim=len(feature_columns))
pricing_model.load_state_dict(torch.load("options_pricing_model.pth"))

pricing_model.eval()

options_data = get_all_options_data()

options_data = options_data.drop_duplicates().dropna()

volume_threshold = 1000
options_data = options_data[options_data['volume'] >= volume_threshold].reset_index(drop=True)

print(options_data)


features = options_data[feature_columns]
print(features)
features = scaler.transform(features)

targets = ((options_data['bid'] + options_data['ask']) / 2).values

with torch.no_grad():
    predictions = pricing_model(torch.tensor(features, dtype=torch.float32))
predictions = predictions.detach().numpy().flatten()

def black_scholes_price(strike, vol, time_to_maturity, moneyness, option_type, spot_price, risk_free_rate, div_yield, 
                        calculation_date, calendar, day_count, bsm_process, is_american=True):
    maturity_date = calculation_date + int(time_to_maturity)
    
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    if is_american:
        exercise = ql.AmericanExercise(calculation_date, maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(bsm_process))
    option_price = option.NPV()
    return option_price

calculation_date = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = calculation_date
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

dividend_yield = 0

prices = []

features = options_data[feature_columns].to_numpy()

for i in range(features.shape[0]):
    print(i)
    datapoint = tuple(features[i])
    strike, vol, time_to_maturity, moneyness, option_type, risk_free_rate = datapoint
    option_type = ql.Option.Call if option_type == 1 else ql.Option.Put
    spot_price = strike * moneyness
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

    calculation_date = ql.Date.todaysDate()
    maturity_date = calculation_date + int(time_to_maturity)

    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, vol, day_count))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
    dividend_yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_yield, day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield_ts, flat_ts, flat_vol_ts)

    price = black_scholes_price(strike, vol, time_to_maturity, moneyness, option_type, spot_price, risk_free_rate, dividend_yield, calculation_date, calendar, day_count, bsm_process, is_american=True)
    prices.append(price)

prices = np.array(prices)


targets = ((options_data['bid'] + options_data['ask']) / 2).values


print(f"Model MSE: {mean_squared_error(targets, predictions)}")
print(f"Black-Scholes MSE: {mean_squared_error(targets, prices)}")




