import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker, period="2y", sequence_length=60):
        """
        Initialize the Stock Predictor
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period (str): Period for historical data ('1y', '2y', '5y', 'max')
            sequence_length (int): Number of days to look back for prediction
        """
        self.ticker = ticker.upper()
        self.period = period
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def fetch_data(self):
        """Fetch historical stock data using yfinance"""
        try:
            print(f"Fetching data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            print(f"Successfully fetched {len(self.data)} days of data")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Clean and preprocess the stock data"""
        if self.data is None:
            print("No data to preprocess. Please fetch data first.")
            return False
            
        # Remove any missing values
        self.data = self.data.dropna()
        
        # Focus on closing prices for prediction
        closing_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(closing_prices)
        
        print("Data preprocessing completed")
        return True
    
    def create_sequences(self, test_size=0.2):
        """Create sequences for LSTM training"""
        if self.scaled_data is None:
            print("No scaled data available. Please preprocess data first.")
            return False
            
        X, y = [], []
        
        # Create sequences
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_index = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_index]
        self.y_train = y[:split_index]
        self.X_test = X[split_index:]
        self.y_test = y[split_index:]
        
        # Reshape for LSTM (samples, time steps, features)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        
        print(f"Created sequences - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return True
    
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM neural network model"""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=lstm_units[0], 
                           return_sequences=True if len(lstm_units) > 1 else False,
                           input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_seq = i < len(lstm_units) - 1
            self.model.add(LSTM(units=lstm_units[i], return_sequences=return_seq))
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='mean_squared_error',
                          metrics=['mae'])
        
        print("LSTM model built successfully")
        print(self.model.summary())
        return True
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """Train the LSTM model"""
        if self.model is None:
            print("No model to train. Please build model first.")
            return False
            
        print("Training model...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=False  # Important for time series data
        )
        
        print("Model training completed")
        return history
    
    def make_predictions(self):
        """Make predictions on test data"""
        if self.model is None:
            print("No trained model available.")
            return None
            
        # Make predictions
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        # Inverse transform to get actual prices
        train_predictions = self.scaler.inverse_transform(train_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        return {
            'train_predictions': train_predictions.flatten(),
            'test_predictions': test_predictions.flatten(),
            'y_train_actual': y_train_actual.flatten(),
            'y_test_actual': y_test_actual.flatten()
        }
    
    def evaluate_model(self, predictions):
        """Evaluate model performance"""
        if predictions is None:
            return None
            
        # Calculate metrics for test set
        rmse = np.sqrt(mean_squared_error(predictions['y_test_actual'], 
                                        predictions['test_predictions']))
        mae = mean_absolute_error(predictions['y_test_actual'], 
                                predictions['test_predictions'])
        
        # Calculate percentage accuracy
        mape = np.mean(np.abs((predictions['y_test_actual'] - predictions['test_predictions']) / 
                             predictions['y_test_actual'])) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Accuracy': 100 - mape
        }
        
        print("\n=== Model Performance Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def create_visualizations(self, predictions):
        """Create interactive Plotly visualizations"""
        if predictions is None:
            return None
            
        # Prepare data for plotting
        train_size = len(predictions['train_predictions'])
        test_size = len(predictions['test_predictions'])
        total_size = train_size + test_size
        
        # Create date index
        dates = self.data.index[-total_size:]
        train_dates = dates[:train_size]
        test_dates = dates[train_size:]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stock Price Prediction', 'Training Loss'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Plot actual vs predicted prices
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=predictions['y_train_actual'],
                mode='lines',
                name='Actual (Train)',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=predictions['train_predictions'],
                mode='lines',
                name='Predicted (Train)',
                line=dict(color='lightblue', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=predictions['y_test_actual'],
                mode='lines',
                name='Actual (Test)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=predictions['test_predictions'],
                mode='lines',
                name='Predicted (Test)',
                line=dict(color='orange', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.ticker} Stock Price Prediction using LSTM',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
        
        return fig
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        if self.model is None:
            print("No trained model available.")
            return None
            
        # Get the last sequence from the data
        last_sequence = self.scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        future_predictions = []
        
        # Predict future prices
        for _ in range(days):
            next_pred = self.model.predict(last_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()
    
    def run_complete_analysis(self, epochs=50, future_days=30):
        """Run the complete stock prediction analysis"""
        print(f"\n=== Stock Market Predictor for {self.ticker} ===\n")
        
        # Step 1: Fetch data
        if not self.fetch_data():
            return None
            
        # Step 2: Preprocess data
        if not self.preprocess_data():
            return None
            
        # Step 3: Create sequences
        if not self.create_sequences():
            return None
            
        # Step 4: Build model
        if not self.build_model():
            return None
            
        # Step 5: Train model
        history = self.train_model(epochs=epochs, verbose=0)
        if not history:
            return None
            
        # Step 6: Make predictions
        predictions = self.make_predictions()
        if predictions is None:
            return None
            
        # Step 7: Evaluate model
        metrics = self.evaluate_model(predictions)
        
        # Step 8: Create visualizations
        fig = self.create_visualizations(predictions)
        
        # Step 9: Predict future
        future_predictions = self.predict_future(days=future_days)
        
        # Display results
        if fig:
            fig.show()
            
        if future_predictions is not None:
            print(f"\n=== Future Price Predictions (Next {future_days} days) ===")
            current_price = self.data['Close'].iloc[-1]
            avg_future_price = np.mean(future_predictions)
            price_change = ((avg_future_price - current_price) / current_price) * 100
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"Average Predicted Price: ${avg_future_price:.2f}")
            print(f"Predicted Change: {price_change:+.2f}%")
            
            # Create future prediction plot
            future_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), 
                                       periods=future_days, freq='D')
            
            fig_future = go.Figure()
            
            # Add historical data (last 60 days)
            recent_data = self.data['Close'].tail(60)
            fig_future.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add future predictions
            fig_future.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    mode='lines+markers',
                    name='Future Predictions',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                )
            )
            
            fig_future.update_layout(
                title=f'{self.ticker} - Future Price Predictions',
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                hovermode='x unified',
                height=500
            )
            
            fig_future.show()
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'future_predictions': future_predictions,
            'model': self.model,
            'data': self.data
        }

# Example usage and main execution
def main():
    """Main function to run the stock predictor"""
    
    # Example with Apple stock
    print("=== Stock Market Predictor Demo ===")
    print("This demo will analyze Apple (AAPL) stock")
    print("You can change the ticker symbol below to analyze different stocks\n")
    
    # Initialize predictor
    ticker = "AAPL"  # Change this to any stock ticker
    predictor = StockPredictor(ticker=ticker, period="2y", sequence_length=60)
    
    # Run complete analysis
    results = predictor.run_complete_analysis(epochs=50, future_days=30)
    
    if results:
        print("\n=== Analysis Complete ===")
        print("Check the interactive plots above for detailed insights!")
        
        # Additional analysis
        print(f"\n=== Stock Information for {ticker} ===")
        print(f"Data Period: {predictor.period}")
        print(f"Total Data Points: {len(predictor.data)}")
        print(f"Date Range: {predictor.data.index[0].date()} to {predictor.data.index[-1].date()}")
        
        # Recent performance
        recent_change = ((predictor.data['Close'].iloc[-1] - predictor.data['Close'].iloc[-30]) / 
                        predictor.data['Close'].iloc[-30]) * 100
        print(f"30-Day Performance: {recent_change:+.2f}%")
        
    else:
        print("Analysis failed. Please check the ticker symbol and try again.")

# Interactive function for custom analysis
def analyze_stock(ticker, period="2y", epochs=50, future_days=30):
    """
    Analyze any stock with custom parameters
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period ('1y', '2y', '5y', 'max')
        epochs (int): Training epochs
        future_days (int): Days to predict into future
    """
    predictor = StockPredictor(ticker=ticker, period=period)
    return predictor.run_complete_analysis(epochs=epochs, future_days=future_days)

if __name__ == "__main__":
    main()
    
    # Uncomment below to analyze different stocks
    # analyze_stock("MSFT", period="1y", epochs=30, future_days=15)
    # analyze_stock("GOOGL", period="2y", epochs=50, future_days=30)
    # analyze_stock("TSLA", period="1y", epochs=40, future_days=20)
