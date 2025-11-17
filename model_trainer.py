import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import data_loader # 匯入我們的 data_loader.py

def create_features(df):
    """從 DataFrame 建立 AI 模型所需的特徵"""
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter # 新增特徵
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)

    # --- 【關鍵修正】 ---
    # 確保 'power_kW' 欄位存在於傳入的 df 中，才能建立 lag feature
    # 我們在 'train_model' 函式中會確保這一點
    if 'power_kW' in df_feat.columns:
        # 建立「昨日同時候的電量」特徵
        # .shift(96) 96 = 24 小時 * 4 筆/小時
        df_feat['lag_1_day'] = df_feat['power_kW'].shift(96)
    else:
        # 這種情況會發生在「預測未來」時，我們會在 app.py 中單獨處理 lag_1_day
        pass 
    
    return df_feat

def train_model(data_df):
    """訓練模型並儲存為 model.pkl"""
    
    print("--- 1. 正在建立特徵 (Features) ---")
    # data_df 傳入時保證有 'power_kW'
    df_train = create_features(data_df)
    
    # 移除因為 lag 特徵而產生的缺失值 (例如第一天的資料)
    df_train = df_train.dropna()
    
    if df_train.empty:
        print("錯誤：建立特徵後沒有剩餘資料可供訓練。")
        return

    print("--- 2. 正在定義特徵 (X) 與目標 (y) ---")
    
    # 這是我們的「模型合約」，特徵必須是這些
    FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'is_weekend', 'lag_1_day']
    TARGET = 'power_kW' # 目標欄位

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]

    print(f"--- 3. 正在訓練模型 (RandomForestRegressor)... ---")
    # 這裡我們用一個簡單的模型作為「假模型」
    # n_estimators=10 讓它跑得快, max_depth=5 防止過擬合
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print(f"--- 4. 訓練完成！正在儲存至 model.pkl ---")
    joblib.dump(model, "model.pkl")
    print("--- model.pkl 已成功儲存！ ---")


# --- 這段是讓您可以「單獨執行」此檔案來產生模型 ---
if __name__ == "__main__":
    print("正在載入完整歷史數據...")
    # 1. 載入資料
    full_data = data_loader.load_all_history_data()
    
    if not full_data.empty:
        # 2. 訓練模型
        train_model(full_data)
    else:
        print("錯誤：沒有載入任何資料，無法訓練模型。")