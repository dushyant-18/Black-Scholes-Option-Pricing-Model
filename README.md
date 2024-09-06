# Black-Scholes Option Pricing Model with Volatility, VaR, and Unit Testing

This repository contains a Python implementation of the Black-Scholes option pricing model for calculating the prices of European call and put options, along with the calculation of historical volatility, Value at Risk (VaR), and associated unit tests. The project makes use of financial data obtained from Yahoo Finance and is structured for efficient testing and validation of the model.

## Features

- **Calculate Historical Volatility**: Fetch historical stock data and compute the annualized historical volatility based on daily returns.
- **Black-Scholes Model**: Calculate the price of European call and put options using the Black-Scholes model.
- **Value at Risk (VaR)**: Compute the VaR for both futures and options (calls and puts) using historical returns.
- **Visualization**: Plot historical stock prices and VaR at different confidence levels.
- **Unit Testing**: Comprehensive unit tests using the `unittest` framework to verify the correctness of volatility calculations, option pricing, and VaR computations.

## Mathematical Formulas

### 1. **Black-Scholes Formula**
The Black-Scholes formula is used to calculate the price of European call and put options.

For a **call option**, the formula is:

C = S_0 . N(d_1) - X . e^{-rT} . N(d_2)

For a **put option**, the formula is:

P = X . e^{-rT} . N(-d_2) - S_0 . N(-d_1)

Where:
- ( S_0 ): Current stock price
- ( X ): Strike price
- ( T ): Time to expiration (in years)
- ( r ): Risk-free interest rate
- ( sigma ): Volatility of the underlying stock
- ( N() ): Cumulative distribution function of the standard normal distribution
- ( d_1 ) and ( d_2 ) are calculated as:

d_1 = {ln(S_0/X) + (r + 0.5 . sigma^2) . T} / {sigma . sqrt(T)}

d_2 = d_1 - sigma . sqrt(T)

### 2. **Value at Risk (VaR) Formula**
Value at Risk (VaR) is used to estimate the potential loss in value of a portfolio over a given time horizon at a specified confidence level.

The formula for VaR at confidence level ( alpha ) is:

VaR_alpha = - Percentile_1 - alpha(returns) . sqrt(T)

Where:
- ( alpha ): Confidence level (e.g., 0.95, 0.99)
- ( returns ): Historical returns of the asset or portfolio
- ( T ): Time horizon (in days)

VaR gives the maximum expected loss over \( T \) days with a confidence level \( \alpha \).

## Dependencies

The project uses the following Python libraries:
- `yfinance`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `unittest` (built-in)
- `unittest.mock` (built-in)

You can install the required libraries with:

```bash
pip install yfinance numpy scipy matplotlib pandas
```

## Usage

### 1. Calculate Historical Volatility and Option Pricing
The `option_pricing_and_var` function combines historical volatility calculation, Black-Scholes option pricing, and VaR computation.

Example usage:

```python
ticker = "AAPL"
option_strike = 202.5
current_stock_price = 222.38
time_to_expiration = 0.5  # 6 months
risk_free_rate = 0.04  # 4%

option_pricing_and_var(ticker, option_strike, current_stock_price, time_to_expiration, risk_free_rate)
```

This function will:
- Fetch stock data for the given ticker.
- Calculate the annualized volatility.
- Compute option prices (call and put) using the Black-Scholes model.
- Calculate VaR for futures and options at multiple confidence levels.
- Display the results with plots.

### 2. Unit Testing
Unit tests are implemented for:
- **Black-Scholes call and put option pricing**.
- **Volatility calculation**.
- **VaR calculation**.

To run the tests, execute:

```bash
python -m unittest discover
```

This will run all the tests in the repository and validate the implementation.

## Code Overview

- **`calculate_volatility(ticker, period='1y')`**: Fetches stock data from Yahoo Finance and computes the annualized historical volatility based on daily returns.
  
- **`black_scholes(S, X, T, r, sigma, option_type='call')`**: Implements the Black-Scholes formula for pricing European options.

- **`calculate_var(returns, confidence_levels=[0.95], time_horizon=1)`**: Calculates the Value at Risk (VaR) for a set of historical returns at different confidence levels.

- **`option_pricing_and_var(ticker, option_strike, current_stock_price, time_to_expiration, risk_free_rate)`**: Main function that integrates volatility calculation, option pricing, and VaR computation, along with generating relevant plots.

### Unit Testing

The `TestOptionPricing` class contains unit tests for:
- **Black-Scholes call and put pricing**.
- **Volatility calculation using mock data**.
- **VaR calculation using sample returns data**.

```python
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
```

This line in the code allows running the tests directly from the script.

This `README.md` now includes the Black-Scholes option pricing formula and the VaR formula, along with a detailed explanation of the project.
