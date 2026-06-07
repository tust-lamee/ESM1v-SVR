import pandas as pd
import joblib


best_svr = joblib.load('9581_svr.pkl')
scaler = joblib.load('9581_svr_scaler.pkl')

unknown_data = pd.read_csv('esm_all_features.csv')  
X_unknown_scaled = scaler.transform(unknown_data)

unknown_predictions = best_svr.predict(X_unknown_scaled)

print("预测的温度值:")
print(unknown_predictions)

pd.DataFrame(unknown_predictions, columns=['Predicted Temperature']).to_csv('predicted_temperatures.csv', index=False)
