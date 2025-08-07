import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def create_ml_features(df):
    """
    Create machine learning features from technical analysis data
    """
    features = df.copy()
    
    # 1. Price-based features
    features['price_change'] = df['close'].pct_change()
    features['high_low_ratio'] = df['high'] / df['low']
    features['body_size'] = abs(df['close'] - df['open'])
    features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    
    # 2. Technical indicators
    # Moving averages
    features['ma_5'] = df['close'].rolling(5).mean()
    features['ma_20'] = df['close'].rolling(20).mean()
    features['ma_50'] = df['close'].rolling(50).mean()
    
    # Price position relative to MAs
    features['above_ma5'] = (df['close'] > features['ma_5']).astype(int)
    features['above_ma20'] = (df['close'] > features['ma_20']).astype(int)
    features['above_ma50'] = (df['close'] > features['ma_50']).astype(int)
    
    # 3. Volatility features
    features['volatility_5'] = df['close'].rolling(5).std()
    features['volatility_20'] = df['close'].rolling(20).std()
    
    # 4. Pattern strength features
    features['engulfing_strength'] = df['engulfing'].rolling(5).sum()
    features['star_strength'] = df['star'].rolling(5).sum()
    
    # 5. Support/Resistance proximity features
    def calculate_level_proximity(df, levels, column='close'):
        """Calculate proximity to nearest support/resistance level"""
        proximity = []
        for i in range(len(df)):
            if levels:
                nearest_level = min(levels, key=lambda x: abs(x - df[column].iloc[i]))
                distance = abs(df[column].iloc[i] - nearest_level) / df[column].iloc[i]
                proximity.append(distance)
            else:
                proximity.append(1.0)  # No levels found
        return proximity
    
    # This will be calculated after support/resistance levels are found
    features['support_proximity'] = 0
    features['resistance_proximity'] = 0
    
    # 6. Time-based features
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    
    # 7. Volume features (if available)
    if 'volume' in df.columns:
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
    else:
        features['volume_ratio'] = 1.0
    
    # 8. RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    features['rsi'] = calculate_rsi(df['close'])
    
    # 9. MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    macd, signal_line, histogram = calculate_macd(df['close'])
    features['macd'] = macd
    features['macd_signal'] = signal_line
    features['macd_histogram'] = histogram
    
    return features

def prepare_ml_data(df, support_levels, resistance_levels):
    """
    Prepare data for machine learning
    """
    # Create features
    features_df = create_ml_features(df)
    
    # Calculate support/resistance proximity
    features_df['support_proximity'] = calculate_level_proximity(df, support_levels, 'low')
    features_df['resistance_proximity'] = calculate_level_proximity(df, resistance_levels, 'high')
    
    # Create target variable (next period's price direction)
    features_df['target'] = np.where(features_df['close'].shift(-1) > features_df['close'], 1, 0)
    
    # Remove NaN values
    features_df = features_df.dropna()
    
    # Select feature columns (exclude target and original OHLCV)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'engulfing', 'star', 'signal']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols]
    y = features_df['target']
    
    return X, y, features_df

def train_ml_model(X, y):
    """
    Train a Random Forest model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    return model, scaler, feature_importance

def predict_signals(df, model, scaler, support_levels, resistance_levels):
    """
    Generate ML-based trading signals
    """
    # Prepare features
    X, _, _ = prepare_ml_data(df, support_levels, resistance_levels)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Create signal column
    df['ml_signal'] = 0
    df.loc[X.index, 'ml_signal'] = predictions
    
    # Add confidence scores
    df['ml_confidence'] = 0.0
    df.loc[X.index, 'ml_confidence'] = np.max(probabilities, axis=1)
    
    return df

if __name__ == "__main__":
    print("ML Feature Engineering Module Ready!")
    print("Use these functions to integrate ML with your technical analysis.") 