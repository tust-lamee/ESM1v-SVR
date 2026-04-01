import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import joblib


pd.set_option('display.float_format', '{:.3f}'.format)
data = pd.read_csv('9579.csv')

first_col_name = data.columns[0]
data[first_col_name] = data[first_col_name].astype(str)

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_dist = {
    'kernel': ['rbf'],
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5]
}

random_search = RandomizedSearchCV(
    SVR(),
    param_dist,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train_scaled, y_train)

cv_results = pd.DataFrame(random_search.cv_results_)

score_cols = [f'split{i}_test_score' for i in range(5)]
param_cols = ['param_C', 'param_gamma', 'param_epsilon', 'param_kernel']
summary_cols = ['mean_test_score', 'std_test_score']

display_cols = param_cols + score_cols + summary_cols
best_idx = cv_results['mean_test_score'].idxmax()

best_svr = random_search.best_estimator_
joblib.dump(best_svr, '9579_svr.pkl')
joblib.dump(scaler, '9579_svr_scaler.pkl')

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = SVR(**random_search.best_params_)
    model.fit(X_tr, y_tr)
    y_pred_val = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    pearson_corr, _ = pearsonr(y_val, y_pred_val)

    fold_metrics.append([mse, rmse, mae, r2, pearson_corr])

    print(f"Fold {fold + 1}: MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}, pearson={pearson_corr:.3f}")

metrics_df = pd.DataFrame(
    fold_metrics,
    columns=['MSE', 'RMSE', 'MAE', 'R²', 'Pearson']
)


mean_vals = metrics_df.mean()
std_vals = metrics_df.std()


summary_table = pd.concat([
    metrics_df,
    pd.DataFrame([mean_vals], index=['Mean']),
    pd.DataFrame([std_vals], index=['Std'])
])


y_pred = best_svr.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R^2: {r2:.3f}")
print(f"pearson correlation: {pearson_corr:.3f}")
plt.figure(figsize=(6, 5))

errors = np.abs(y_test - y_pred)
outlier_idx_in_test = np.argmax(errors)

original_index = X_test.index[outlier_idx_in_test]

main_color = '#009E73'
line_color = '#E69F00'
edge_color = '#333333'

plt.scatter(y_test, y_pred, alpha=0.75, color=main_color,
            edgecolors=edge_color, linewidth=0.5, s=65, label='Predicted vs True')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color=line_color, linestyle='--', linewidth=2, label='Ideal Fit')

plt.text(
    min(y_test) + (max(y_test) - min(y_test)) * 0.05,
    max(y_pred) - (max(y_pred) - min(y_pred)) * 0.25,
    f'$R^2$ = {r2:.3f}',
    fontsize=13,
    color='black',
    fontname='Times New Roman',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
)

plt.legend(frameon=False, fontsize=11, prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('SVR_True_vs_Predicted_300dpi.png', dpi=300, bbox_inches='tight')
plt.show()