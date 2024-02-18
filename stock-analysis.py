
pip install yfinance
pip install getFamaFrenchFactors

import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff

#ALK-Abelló A/S 

ticker = 'alk-b.co'

# perioada 2007-2009
start = '2007-1-01'
end = '2009-12-31'

# perioada 2020-2022
start = '2020-1-01'
end = '2022-12-31'

# perioada 2007-2022
start = '2007-1-01'
end = '2022-12-31'

stock_data = yf.download(ticker, start, end)

ff3_monthly = gff.famaFrench3Factor(frequency='m')
ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff3_monthly.set_index('Date', inplace=True)

stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
stock_returns.name = "Month_Rtn"
ff_data = ff3_monthly.merge(stock_returns,on='Date')

"""
CAPM
"""

rf = ff_data['RF'].mean()
market_premium = ff_data['Mkt-RF'].mean()

X = ff_data['Mkt-RF']
y =  ff_data['Month_Rtn'] - ff_data['RF']
c = sm.add_constant(X)
capm_model = sm.OLS(y, c)
result = capm_model.fit()
print(result.summary())
intercept, beta = result.params

expected_return = rf + beta*market_premium
print("CAPM Expected monthly returns: " + str(expected_return))
yearly_return = expected_return * 12
print("CAPM Expected yearly returns: " + str(yearly_return))

"""
Fama-French 3 Factors Model
"""

X = ff_data[['Mkt-RF', 'SMB', 'HML']]
y = ff_data['Month_Rtn'] - ff_data['RF']
X = sm.add_constant(X)
ff_model = sm.OLS(y, X).fit()
print(ff_model.summary())
intercept, b1, b2, b3 = ff_model.params

rf = ff_data['RF'].mean()
market_premium = ff3_monthly['Mkt-RF'].mean()
size_premium = ff3_monthly['SMB'].mean()
value_premium = ff3_monthly['HML'].mean()

expected_monthly_return = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium 
expected_yearly_return = expected_monthly_return * 12
print("Expected yearly return: " + str(expected_yearly_return))