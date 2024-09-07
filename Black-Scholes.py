import yfinance as yf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import unittest
from unittest.mock import patch


# Function to fetch data and calculate historical volatility
def calculate_volatility(ticker, period='1y'):
    """
    Calculate annualized historical volatility and daily returns for a given stock ticker.

    Parameters:
    ticker : str : Stock ticker symbol
    period : str : Historical period to fetch data for (default is 1 year)

    Returns:
    annualized_volatility : float : Annualized historical volatility
    daily_returns : pd.Series : Daily returns for the stock
    hist : pd.DataFrame : Historical stock data
    """
    # Fetch historical data from Yahoo Finance
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period=period)

    # Calculate daily returns
    hist['Daily Return'] = np.log(hist['Close'] / hist['Close'].shift(1))

    # Calculate the standard deviation of daily returns (daily volatility)
    daily_volatility = np.std(hist['Daily Return'].dropna())

    # Annualize the volatility
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_volatility, hist['Daily Return'].dropna(), hist


# Function to compute Black-Scholes option pricing
def black_scholes(S, X, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price for European call or put options.

    Parameters:
    S : float : Current stock price
    X : float : Strike price
    T : float : Time to expiration (in years)
    r : float : Risk-free interest rate (annual)
    sigma : float : Volatility of the underlying stock (annual)
    option_type : str : 'call' or 'put' to calculate the respective option price

    Returns:
    option_price : float : Black-Scholes price of the option
    """
    # Calculate d1 and d2
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate option price
    if option_type == 'call':
        option_price = S * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        option_price = X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price


# Function to calculate VaR based on historical returns
def calculate_var(returns, confidence_levels=[0.95], time_horizon=1):
    """
    Calculate the Value at Risk (VaR) at given confidence levels.

    Parameters:
    - returns: historical returns (daily)
    - confidence_levels: list of confidence levels for VaR (default is [0.95, 0.99, 0.975])
    - time_horizon: the number of days for which VaR is calculated

    Returns:
    - A dictionary with confidence levels as keys and VaR values as values
    """
    var_dict = {}

    for confidence_level in confidence_levels:
        # Calculate the percentile for the given confidence level
        var = np.percentile(returns, (1 - confidence_level) * 100) * np.sqrt(time_horizon)
        # VaR should be positive to reflect risk
        var_dict[f'{confidence_level * 100:.0f}%'] = -var

    return var_dict


# Main function to fetch data, compute Black-Scholes prices, and calculate VaR
def option_pricing_and_var(ticker, option_strike, current_stock_price, time_to_expiration, risk_free_rate):
    """
    Combines option pricing using Black-Scholes and risk management using VaR.

    Parameters:
    ticker : str : Stock ticker symbol
    option_strike : float : Strike price of the option
    current_stock_price : float : Current stock price
    time_to_expiration : float : Time to expiration in years (e.g., 0.5 for 6 months)
    risk_free_rate : float : Risk-free interest rate (e.g., 0.04 for 4%)

    Displays volatility, Black-Scholes prices, and VaR.
    """
    # Calculate volatility and returns
    sigma, daily_returns, hist = calculate_volatility(ticker)

    # Plot the historical stock data
    plt.figure(figsize=(10, 6))
    plt.plot(hist.index, hist['Close'], label='Close Price')
    plt.title(f"Historical Close Price for {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    print(f"Annualized Historical Volatility for {ticker}: {sigma:.4f}")

    # Calculate option prices using the Black-Scholes model
    call_price = black_scholes(S=current_stock_price, X=option_strike, T=time_to_expiration, r=risk_free_rate,
                               sigma=sigma, option_type='call')
    put_price = black_scholes(S=current_stock_price, X=option_strike, T=time_to_expiration, r=risk_free_rate,
                              sigma=sigma, option_type='put')

    print(f"Call Option Price: {call_price:.4f}")
    print(f"Put Option Price: {put_price:.4f}")

    # Calculate VaR for futures (stock itself)
    confidence_levels = [0.95, 0.99, 0.975]
    var_dict_futures = calculate_var(daily_returns, confidence_levels=confidence_levels)

    # Approximate option returns by scaling stock returns by option prices
    call_returns = daily_returns * (call_price / current_stock_price)
    put_returns = daily_returns * (put_price / current_stock_price)

    # Calculate VaR for options (calls and puts)
    var_dict_call = calculate_var(call_returns, confidence_levels=confidence_levels)
    var_dict_put = calculate_var(put_returns, confidence_levels=confidence_levels)

    # Display VaR results
    print(f"\nValue at Risk (VaR) at different confidence levels:")
    print(f"VaR Futures:\n{pd.Series(var_dict_futures)}")
    print(f"VaR Call:\n{pd.Series(var_dict_call)}")
    print(f"VaR Put:\n{pd.Series(var_dict_put)}")

    # Plot VaR results
    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    index = np.arange(len(confidence_levels))

    plt.bar(index - bar_width, list(var_dict_futures.values()), bar_width, label='Futures', color='blue')
    plt.bar(index, list(var_dict_call.values()), bar_width, label='Call', color='green')
    plt.bar(index + bar_width, list(var_dict_put.values()), bar_width, label='Put', color='red')

    plt.xlabel('Confidence Level')
    plt.ylabel('Value at Risk (VaR)')
    plt.title(f'VaR at Different Confidence Levels for {ticker}')
    plt.xticks(index, list(var_dict_futures.keys()))
    plt.legend()
    plt.show()

# Example usage
ticker = "AAPL"  # Use Yahoo Finance's AAPL ticker
option_strike = 202.5  # Example strike price
current_stock_price = 222.38  # Example current stock price
time_to_expiration = 0.5  # 6 months
risk_free_rate = 0.04  # 4%

# Call the main function to display results
option_pricing_and_var(ticker, option_strike, current_stock_price, time_to_expiration, risk_free_rate)