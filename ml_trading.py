import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

class MLTradingSystem:
    def __init__(self, symbol="AAPL", period="5y", sequence_length=60):
        self.symbol = symbol
        self.period = period
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def load_data(self):
        """Load and prepare data"""
        print(f"Loading data for {self.symbol}...")
        
        # Download data
        df = yf.download(self.symbol, period=self.period)
        df.reset_index(inplace=True)
        
        # Clean data
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Create features
        df = self.create_features(df)
        
        print(f"Data loaded: {len(df)} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def create_features(self, df):
        """Create technical features from raw price data"""
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['body_size'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Moving averages
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        
        # Price position relative to MAs
        df['above_ma5'] = (df['Close'] > df['ma_5']).astype(int)
        df['above_ma20'] = (df['Close'] > df['ma_20']).astype(int)
        df['above_ma50'] = (df['Close'] > df['ma_50']).astype(int)
        
        # Volatility features
        df['volatility_5'] = df['Close'].rolling(5).std()
        df['volatility_20'] = df['Close'].rolling(20).std()
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['Close'])
        
        # MACD
        macd, signal, histogram = self.calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def create_sequences(self, df, target_column='Close'):
        """Create sequences for LSTM"""
        # Select feature columns
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'price_change', 'high_low_ratio', 'body_size',
                          'upper_shadow', 'lower_shadow', 'ma_5', 'ma_20', 'ma_50',
                          'above_ma5', 'above_ma20', 'above_ma50',
                          'volatility_5', 'volatility_20', 'volume_ratio',
                          'rsi', 'macd', 'macd_signal', 'macd_histogram',
                          'hour', 'day_of_week', 'month']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            # Input sequence
            X.append(scaled_data[i-self.sequence_length:i])
            
            # Target: Next period return
            current_price = df[target_column].iloc[i]
            next_price = df[target_column].iloc[i+1] if i+1 < len(df) else current_price
            return_rate = (next_price - current_price) / current_price
            
            # Binary classification: 1 if profitable, 0 if not
            y.append(1 if return_rate > 0.01 else 0)  # 1% threshold
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_model(self, input_shape):
        """Build CNN model for pattern recognition"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, model_type='lstm'):
        """Train the model"""
        # Load and prepare data
        df = self.load_data()
        X, y = self.create_sequences(df)
        
        print(f"Created {len(X)} sequences")
        print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        if model_type == 'lstm':
            self.model = self.build_lstm_model((X.shape[1], X.shape[2]))
        elif model_type == 'cnn':
            self.model = self.build_cnn_model((X.shape[1], X.shape[2]))
        
        print(f"Model architecture ({model_type.upper()}):")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        self.evaluate_model(X_test, y_test, history)
        
        return history
    
    def evaluate_model(self, X_test, y_test, history):
        """Evaluate model performance"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_binary)
        print(cm)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot predictions vs actual
        self.plot_predictions(y_test, y_pred)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred):
        """Plot predictions vs actual"""
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.subplot(1, 2, 1)
        plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predictions')
        plt.scatter(range(len(y_true)), y_true, alpha=0.5, label='Actual')
        plt.title('Predictions vs Actual')
        plt.xlabel('Sample')
        plt.ylabel('Probability/Class')
        plt.legend()
        
        # Plot prediction distribution
        plt.subplot(1, 2, 2)
        plt.hist(y_pred[y_true == 0], alpha=0.5, label='Actual Loss', bins=20)
        plt.hist(y_pred[y_true == 1], alpha=0.5, label='Actual Profit', bins=20)
        plt.title('Prediction Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_signals(self, df=None):
        """Generate trading signals"""
        if df is None:
            df = self.load_data()
        
        X, _ = self.create_sequences(df)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Create signal DataFrame
        signal_df = df.iloc[self.sequence_length:].copy()
        signal_df['prediction'] = predictions.flatten()
        signal_df['signal'] = (predictions > 0.5).astype(int)
        signal_df['confidence'] = np.maximum(predictions, 1 - predictions).flatten()
        
        return signal_df
    
    def backtest(self, signal_df, initial_capital=10000):
        """Simple backtest"""
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(len(signal_df)):
            current_price = signal_df['Close'].iloc[i]
            prediction = signal_df['prediction'].iloc[i]
            confidence = signal_df['confidence'].iloc[i]
            
            # Trading logic: Buy if prediction > 0.7 and confidence > 0.6
            if prediction > 0.7 and confidence > 0.6 and position == 0:
                # Buy
                position = capital / current_price
                capital = 0
                trades.append({
                    'date': signal_df.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'confidence': confidence
                })
            
            elif position > 0 and prediction < 0.3:
                # Sell
                capital = position * current_price
                position = 0
                trades.append({
                    'date': signal_df.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'confidence': confidence
                })
        
        # Final position
        if position > 0:
            capital = position * signal_df['Close'].iloc[-1]
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        print(f"\nBacktest Results:")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        return trades, total_return

# Example usage
if __name__ == "__main__":
    # Initialize system
    ml_system = MLTradingSystem(symbol="AAPL", period="5y", sequence_length=60)
    
    # Train LSTM model
    print("Training LSTM model...")
    history = ml_system.train_model(model_type='lstm')
    
    # Generate signals
    print("\nGenerating trading signals...")
    signal_df = ml_system.predict_signals()
    
    # Backtest
    print("\nRunning backtest...")
    trades, return_pct = ml_system.backtest(signal_df)
    
    print(f"\nML Trading System Complete!")
    print(f"Symbol: {ml_system.symbol}")
    print(f"Period: {ml_system.period}")
    print(f"Sequence Length: {ml_system.sequence_length}")

