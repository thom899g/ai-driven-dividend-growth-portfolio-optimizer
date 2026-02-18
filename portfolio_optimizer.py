from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioOptimizer:
    def __init__(self):
        self.portfolio = {}
        
    def calculate_holding_period(self, stock_data: pd.DataFrame) -> Dict[str, int]:
        """
        Calculates optimal holding periods for each stock using machine learning predictions.
        
        Args:
            stock_data (pd.DataFrame): DataFrame containing stock data with predictions.
            
        Returns:
            Dict[str, int]: Dictionary mapping tickers to their optimal holding periods.
        """
        if not isinstance(stock_data, pd.DataFrame):
            raise TypeError("stock_data must be a pandas DataFrame.")
            
        # Assuming 'predicted_duration' is a column in stock_data
        holding_periods = {}
        for ticker in stock_data.index.get_level_values(1).unique().tolist():
            predicted_days = int(np.round(stock_data.loc[ticker]['predicted_duration']))
            holding_periods[ticker] = max(predicted_days, 30)  # Minimum holding period
            
        logging.info("Optimal holding periods calculated.")
        return holding_periods

    def optimize_portfolio(self, undervalued_stocks: List[str], current_value: float) -> Dict:
        """
        Optimizes the portfolio based on risk tolerance and investment goals.
        
        Args:
            undervalued_stocks (List[str]): List of undervalued stock tickers.
            current_value (float): Current portfolio value.
            
        Returns:
            Dict: Optimized portfolio allocation.
        """
        # Simple optimization logic - equal weight for each stock
        num_stocks = len(undervalued_stocks)
        allocation = {'cash': 100 - (current_value / num_stocks),
                     **{ticker: current_value / num_stocks for ticker in undervalued_stocks}}
        
        logging.info(f"Portfolio optimized with {len(allocation)} positions.")
        return allocation