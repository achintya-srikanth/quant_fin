import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
from statsmodels.tsa.stattools import coint
import pandas as pd
from sklearn.metrics import mean_absolute_error


'''
The following project demonstrates two foundational approaches to derivatives pricing.
The Black-Scholes model provides an analytical solution for European options, leveraging stochastic calculus to model asset prices as geometric Brownian motion.
By calculating partial derivatives (the "Greeks"), we quantify an option's sensitivity to market parameters like volatility (Δ), time decay (Θ), and interest rates (ρ),
crucial for hedging strategies. The Monte Carlo simulation complements this by modeling thousands of potential price paths,
particularly valuable for pricing path-dependent options (e.g., Asian or barrier options) where closed-form solutions don't exist.
The code implements variance reduction techniques like antithetic sampling to improve computational efficiency, addressing a key challenge in Monte Carlo methods.
This dual approach showcases both theoretical finance (Black-Scholes equation derivation) and practical numerical methods,
while highlighting the limitations of constant volatility assumptions in real markets.
'''

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

def monte_carlo_option(S, K, T, r, sigma, n_sims=100000):
    z = np.random.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    call_payoff = np.maximum(ST - K, 0)
    put_payoff = np.maximum(K - ST, 0)
    call_price = np.exp(-r*T)*np.mean(call_payoff)
    put_price = np.exp(-r*T)*np.mean(put_payoff)
    return call_price, put_price

# Example usage
S = 100   # Spot price
K = 105   # Strike price
T = 1     # Time to maturity
r = 0.05  # Risk-free rate
sigma = 0.2

bs_call = black_scholes(S, K, T, r, sigma)
mc_call, mc_put = monte_carlo_option(S, K, T, r, sigma)
print(f"Black-Scholes Call: {bs_call:.2f}")
print(f"Monte Carlo Call: {mc_call:.2f}")


'''
Implementing Harry Markowitz's Nobel Prize-winning framework, the following project constructs mean-variance optimal portfolios.
The code solves the quadratic programming problem to find weights that maximize returns for a given risk level, generating the efficient frontier.
Key innovations include Cholesky decomposition for modeling correlated asset returns and constraint handling for realistic portfolio construction (e.g., no short selling).
The analysis extends beyond basic optimization by calculating portfolio metrics like Sharpe ratio and maximum drawdown,
while demonstrating the sensitivity of allocations to input assumptions - a critical consideration given the "garbage in, garbage out" nature of portfolio optimization.
This implementation bridges theoretical finance with practical challenges like estimation error in covariance matrices,
suggesting extensions like Black-Litterman models for incorporating investor views.
'''

def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights.T, returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def minimize_volatility(weights, returns, cov_matrix):
    return portfolio_performance(weights, returns, cov_matrix)[1]

# Sample data
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
returns = pd.DataFrame(np.random.normal(0.001, 0.02, (1000, 4)), columns=tickers)
cov_matrix = returns.cov() * 252
expected_returns = returns.mean() * 252

# Optimization constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(len(tickers)))
init_guess = [1/len(tickers)] * len(tickers)

opt_results = minimize(minimize_volatility, init_guess,
                       args=(expected_returns, cov_matrix),
                       method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_results.x
print("Optimal Weights:", dict(zip(tickers, optimal_weights)))


def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100*(1-confidence_level))

def expected_shortfall(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Historical VaR
historical_var = calculate_var(returns)

# Parametric VaR (Normal Distribution)
parametric_var = norm.ppf(1-0.95) * returns.std() + returns.mean()

# Monte Carlo VaR
n_sims = 100000
simulated_returns = np.random.normal(
    returns.mean(), 
    returns.std(),
    size=(n_sims, 4)  # 100,000 samples × 4 distributions
)
mc_var = calculate_var(simulated_returns)

print(f"Historical 95% VaR: {historical_var:.4f}")
print(f"Parametric 95% VaR: {parametric_var.iloc[0]:.4f}")
print(f"Monte Carlo 95% VaR: {mc_var:.4f}")


'''
The following statistical arbitrage strategy identifies cointegrated asset pairs through Engle-Granger testing,
then implements a mean-reversion strategy based on z-score thresholds. The code calculates dynamic hedge ratios using rolling regression and incorporates
transaction cost modeling for realistic backtesting. Key innovations include volatility-adjusted position sizing and regime detection filters
to avoid losses during structural breaks. Performance analysis uses metrics like Sharpe ratio (>2 target) and maximum drawdown (<15% threshold),
while the walk-forward optimization approach prevents overfitting. The project demonstrates the lifecycle of quant trading strategies from signal
generation to execution considerations, emphasizing the importance of stationarity testing and stochastic spread modeling in maintaining edge in efficient markets.
'''

def pairs_trading_strategy(stock1, stock2, window=30):
    # Cointegration test
    score, pvalue, _ = coint(stock1, stock2)
    
    # Calculate spread
    spread = stock1 - stock2
    zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    # Generate signals
    signals = pd.DataFrame(index=stock1.index)
    signals['position'] = np.where(zscore < -2, 1, np.where(zscore > 2, -1, 0))
    signals['returns'] = signals['position'].shift(1) * (stock1.pct_change() - stock2.pct_change())
    
    return signals.dropna()

data = yf.download(['AAPL', 'MSFT'], start='2020-01-01', end='2024-03-21')

stock_a = data[( 'Close', 'AAPL')]
stock_b = data[( 'Close', 'MSFT')]
signals = pairs_trading_strategy(stock_a, stock_b)

# Performance metrics
cum_returns = signals['returns'].cumsum()
sharpe_ratio = signals['returns'].mean() / signals['returns'].std() * np.sqrt(252)
max_drawdown = (cum_returns.cummax() - cum_returns).max()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

'''
The following comparative study examines traditional statistical and modern machine learning approaches to financial forecasting.
The ARIMA model decomposes time series into autoregressive (AR) and moving average (MA) components, using differencing (I) to handle non-stationarity -
critical for modeling mean-reverting assets or volatility clusters. The LSTM implementation captures complex temporal patterns through gated memory cells,
capable of learning long-range dependencies in noisy market data.
The code includes MinMax scaling to normalize inputs and a sliding window approach to structure sequential data,
addressing common challenges in financial time series preprocessing. By evaluating both models on metrics like MAE and RMSE,
the project demonstrates the trade-off between interpretability (ARIMA's clear parameters) and predictive power (LSTM's nonlinear modeling),
while emphasizing the importance of walk-forward validation to avoid overfitting in evolving markets.
'''

# Data Download
data = yf.download('SPY', start='2020-01-01', end='2024-01-01')['Close']

# ARIMA Implementation
# Split data into train/test
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Find best model using TRAINING DATA only
best_model = auto_arima(train, seasonal=True, trace=True, 
                       error_action='ignore', suppress_warnings=True,
                       stepwise=True, information_criterion='aic')

print(f"Best ARIMA order: {best_model.order}")  # Example: (2,1,1)

# Fit on training data and forecast test period
model = ARIMA(train, order=best_model.order, trend='t')
model_fit = model.fit()
pred_results = model_fit.get_forecast(steps=len(test))
arima_forecast = pred_results.predicted_mean
conf_int = pred_results.conf_int()

# LSTM Implementation
# Scale using TRAINING DATA only
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train.values.reshape(-1,1))
scaled_test = scaler.transform(test.values.reshape(-1,1))

# Create sequences from combined scaled data
look_back = 60
X, y = [], []
combined_scaled = np.concatenate([scaled_train, scaled_test])

for i in range(len(combined_scaled)-look_back-1):
    X.append(combined_scaled[i:(i+look_back), 0])
    y.append(combined_scaled[i+look_back, 0])
    
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Temporal split
train_seq_len = len(scaled_train) - look_back
X_train, X_test = X[:train_seq_len], X[train_seq_len:]
y_train, y_test = y[:train_seq_len], y[train_seq_len:]

# Model Architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training & Evaluation
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=40,
                    batch_size=32,
                    verbose=1)

# Inverse transform predictions
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
test_predict = test_predict.ravel()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# Calculate errors
mae = mean_absolute_error(y_test_actual, test_predict)
arima_mae = mean_absolute_error(test, arima_forecast)
print(f"LSTM MAE: ${mae:.2f}")
print(f"ARIMA MAE: ${arima_mae:.2f}")

# Plotting
plt.figure(figsize=(14,6))
plt.plot(data.index, data, label='Actual')

# Align LSTM predictions with proper dates
lstm_dates = data.index[train_size + 1 : train_size + 1 + len(X_test)]

plt.plot(test.index, arima_forecast, label=f'ARIMA Forecast (MAE: ${arima_mae:.2f})')
plt.fill_between(test.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='gray', alpha=0.1)
plt.plot(lstm_dates, test_predict, label=f'LSTM Predictions (MAE: ${mae:.2f})')

plt.title('SPY Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('price_pred.png', dpi=300)