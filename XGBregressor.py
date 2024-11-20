import pandas as pd
import numpy as np

# 設定隨機種子，方便結果重現
np.random.seed(42)

# 生成一個簡單的虛擬房價數據集
data_size = 1000
X1 = np.random.normal(50, 10, data_size)  # 面積（平方公尺）
X2 = np.random.randint(1, 5, data_size)   # 房間數量
X3 = np.random.randint(1, 30, data_size)  # 樓層
noise = np.random.normal(0, 5, data_size) # 噪音

# 房價的生成規則
y = (X1 * 3000 + X2 * 50000 + X3 * 1000 + noise)

# 將數據存入 DataFrame
df = pd.DataFrame({
    'Area': X1,
    'Rooms': X2,
    'Floor': X3,
    'Price': y
})

print(df.head())

from sklearn.model_selection import train_test_split

# 分離特徵和目標變數
X = df.drop('Price', axis=1)
y = df['Price']

# 分割數據集，80% 為訓練集，20% 為測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")


from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# 定義你想要調整的參數範圍
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.3, 0.7, 1.0],
    'gamma': [0, 0.1, 0.5],
    'min_child_weight': [1, 5, 10]
}

# 初始化 XGBoost 回歸器
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# 使用 GridSearchCV 搜索最佳參數
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)

# 執行調參
grid_search.fit(X_train, y_train)


# 輸出最佳參數
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")


# 使用最終的模型和參數
xg_reg_final = xgb.XGBRegressor(**grid_search.best_params_)

# 進行交叉驗證
cv_results = cross_val_score(xg_reg_final, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# 輸出平均的 MSE
print(f"Cross-validation Mean Squared Error: {-cv_results.mean()}")

# 訓練最終模型
xg_reg_final.fit(X_train, y_train)

# 預測測試集
final_predictions = xg_reg_final.predict(X_test)

# 計算最終均方誤差
final_mse = mean_squared_error(y_test, final_predictions)
print(f"Final Mean Squared Error: {final_mse}")
