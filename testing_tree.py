from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import glob
import os


df = pd.read_csv("data/my_trading_signals_apple.csv")

df = df.dropna(subset=['target_category', 'signal'])

dfc = df[df['target_category'].isin([0, 1, 2])].copy()

dfc['label'] = dfc['target_category']

feature_cols = ['open', 'high', 'low', 'close', 'engulfing', 'star', 'rsi', 'ema_20']
X = dfc[feature_cols]
y = dfc['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=12,
    max_depth=12
)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

joblib.dump(model, 'model.pkl')
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
