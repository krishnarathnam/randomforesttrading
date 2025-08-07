from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv("my_trading_signals_apple.csv")

dfc = df[df['target_category'].isin([1, 2])].copy()

dfc['label'] = dfc.apply(
    lambda row: 1 if (row['signal'] == 1 and row['target_category'] == 1) or
                      (row['signal'] == 2 and row['target_category'] == 2) else 0,
    axis=1
)

dfc.drop(['target_category', 'target_amount'], axis=1, inplace=True)
feature_cols = ['open', 'high', 'low', 'close', 'signal', 'engulfing', 'star']
X = dfc[feature_cols]
y = dfc['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

joblib.dump(model, 'model.pkl')
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

