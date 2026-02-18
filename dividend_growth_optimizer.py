import yfinance as yf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DividendGrowthOptimizer:
    def __init__(self):
        self.data = None
        self.model = None
        self.tickers = []

    def fetch_data(self, tickers: list, start_date: str, end_date: str) -> None:
        """
        Fetches historical data for given tickers from Yahoo Finance.
        
        Args:
            tickers (list): List of stock ticker symbols.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        try:
            self.data = yf.download(tickers, start=start_date, end=end_date)
            logging.info(f"Fetched data for tickers: {tickers}")
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")

    def preprocess_data(self) -> DataFrame:
        """
        Preprocesses fetched data by handling missing values and adding dividend metrics.
        
        Returns:
            DataFrame: Processed stock data with added metrics.
        """
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data first.")
            
        # Handle missing values
        self.data.dropna(inplace=True)
        
        # Calculate dividend yield and payout ratio
        self.data['Dividend Yield'] = (self.data['dividends'] / self.data['Adj Close']) * 100
        self.data['Payout Ratio'] = (self.data['dividends'] / self.data['earnings'])
        
        logging.info("Preprocessing completed with added metrics.")
        return self.data

    def detect_undervalued_stocks(self) -> list:
        """
        Identifies undervalued stocks based on dividend metrics.
        
        Returns:
            list: List of undervalued stock tickers.
        """
        if self.data is None:
            raise ValueError("Data not fetched or preprocessed. Call fetch_data and preprocess_data first.")
            
        # Simple heuristic: consider stocks with high dividend yield and reasonable payout ratio
        criteria = (self.data['Dividend Yield'] > 5) & (self.data['Payout Ratio'] < 0.6)
        
        undervalued_stocks = self.data[criteria].index.get_level_values(1).unique().tolist()
        
        logging.info(f"Detected {len(undervalued_stocks)} undervalued stocks.")
        return undervalued_stocks

    def train_model(self, features: list, target: str) -> None:
        """
        Trains a machine learning model to predict optimal holding periods.
        
        Args:
            features (list): List of feature columns.
            target (str): Target column for prediction.
        """
        if self.data is None:
            raise ValueError("Data not fetched or preprocessed. Call fetch_data and preprocess_data first.")
            
        # Split data into training and testing sets
        X = self.data[features]
        y = self.data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Initialize and train the model
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        
        logging.info(f"Model trained with {len(X_train)} samples.")