import streamlit as st
import pandas as pd
import joblib
import os
import json
import time
import requests
from datetime import datetime, timedelta

# --- 設定 Pantry Cloud ID (從模型訓練程式碼取得) ---
POWER_PANTRY_ID = "6a2e85f5-4af4-4efd-bb9f-c5604fe8475e"
TARGET_YEARS = [2023, 2024, 2025, 2026] # 設定要抓取的年份範圍

# --- 1. Lottie 動畫載入函式 ---
@st.cache_data
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None

# --- 2. 核心快取功能 (Caching) ---
@st.cache_resource
def load_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        return None

# --- 輔助函式：抓取單一 Basket (移植自模型訓練程式碼) ---
def fetch_basket(pantry_id: str, basket_name: str, max_retries: int = 3):
    url = f"https://getpantry.cloud/apiv1/pantry/{pantry_id}/basket/{basket_name}"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                try:
                    return r.json()
                except:
                    return None
            if r.status_code == 404: 
                return None # 該季度資料不存在 (例如未來的時間)
            if r.status_code == 429: # Too Many Requests
                time.sleep(1.0 * (attempt + 1))
                continue
        except:
            time.sleep(1.0)
    return None

# --- 核心數據載入函式 (改為雲端多季度抓取) ---
@st.cache_data(ttl=300) # 5分鐘快取
def load_data():
    """
    從 Pantry Cloud 迴圈抓取多個年份與季度的資料，並合併清洗。
    """
    all_records = []
    
    # 顯示進度條，避免使用者以為當機
    progress_text = "正在從雲端同步歷史數據..."
    my_bar = st.progress(0, text=progress_text)
    
    total_steps = len(TARGET_YEARS) * 4
    current_step = 0
    
    for year in TARGET_YEARS:
        for q in range(1, 5): # Q1 ~ Q4
            basket_name = f"{year}-q{q}"
            
            # 更新進度條
            current_step += 1
            my_bar.progress(current_step / total_steps, text=f"{progress_text} ({basket_name})")
            
            # 抓取資料
            data = fetch_basket(POWER_PANTRY_ID, basket_name)
            
            if data and "data" in data:
                # 這裡假設 "data" 裡面是一個 list of dicts
                all_records.extend(data["data"])
            
            # 稍微休息避免觸發 API 限制
            time.sleep(0.05)
            
    my_bar.empty() # 載入完成後隱藏進度條

    if not all_records:
        st.error("❌ 無法從雲端取得任何數據，請檢查網路或 Pantry ID。")
        return pd.DataFrame()

    # --- 資料清洗與轉換 ---
    try:
        df = pd.DataFrame(all_records)
        
        # 1. 處理時間欄位 (模型組用的是 'full_timestamp' 或 'date'+'time')
        if "full_timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["full_timestamp"], errors="coerce")
        elif "date" in df.columns and "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        else:
            # 嘗試自動尋找
            for col in ['created_at', 'Time', 'time']:
                if col in df.columns:
                    df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
                    break
        
        # 2. 處理電力欄位 (模型組用的是 'power')
        if "power" in df.columns:
            df.rename(columns={"power": "power_kW"}, inplace=True)
        
        # 3. 格式標準化
        df["power_kW"] = pd.to_numeric(df["power_kW"], errors="coerce")
        df.dropna(subset=["timestamp", "power_kW"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        # 去除重複 (因為有些季度交接處可能有重複數據)
        df = df[~df.index.duplicated(keep='first')]
        
        return df[['power_kW']]

    except Exception as e:
        st.error(f"資料解析失敗: {e}")
        return pd.DataFrame()

# --- 3. 電價計算邏輯 (保持不變) ---
PROGRESSIVE_RATES = [
    (120, 1.68, 1.68), (210, 2.45, 2.16), (170, 3.70, 3.03),
    (200, 5.04, 4.14), (300, 6.24, 5.07), (float('inf'), 8.46, 6.63)
]
TOU_RATES_DATA = {
    'basic_fee_monthly': 75.0, 'surcharge_kwh_threshold': 2000.0, 'surcharge_rate_per_kwh': 0.99,
    'rates': {'summer': {'peak': 4.71, 'off_peak': 1.85}, 'nonsummer': {'peak': 4.48, 'off_peak': 1.78}}
}

def calculate_progressive_cost(total_kwh_month, is_summer):
    cost = 0
    kwh_remaining = total_kwh_month
    rate_index = 1 if is_summer else 2
    for (bracket_kwh, *rates) in PROGRESSIVE_RATES:
        rate = rates[rate_index - 1]
        if kwh_remaining <= 0: break
        kwh_in_bracket = min(kwh_remaining, bracket_kwh)
        cost += kwh_in_bracket * rate
        kwh_remaining -= kwh_in_bracket
    return cost

def get_tou_details(timestamp):
    is_summer = (timestamp.month >= 6) and (timestamp.month <= 9)
    is_weekend = timestamp.dayofweek >= 5
    hour = timestamp.hour
    category = 'off_peak'
    if not is_weekend:
        if is_summer:
            if 9 <= hour < 24: category = 'peak'
        else:
            if (6 <= hour < 11) or (14 <= hour < 24): category = 'peak'
    season = 'summer' if is_summer else 'nonsummer'
    rate = TOU_RATES_DATA['rates'][season][category]
    return category, rate, is_summer

@st.cache_data
def analyze_pricing_plans(df_period):
    df_analysis = df_period.copy()
    tou_details = df_analysis.index.map(get_tou_details)
    df_analysis['tou_category'] = [cat for cat, rate, season in tou_details]
    df_analysis['tou_rate'] = [rate for cat, rate, season in tou_details]
    df_analysis['kwh'] = df_analysis['power_kW'] * 0.25
    df_analysis['tou_flow_cost'] = df_analysis['kwh'] * df_analysis['tou_rate']
    monthly_tou = df_analysis.resample('MS').agg(kwh=('kwh', 'sum'), flow_cost=('tou_flow_cost', 'sum'))
    monthly_tou['basic_fee'] = TOU_RATES_DATA['basic_fee_monthly']
    threshold = TOU_RATES_DATA['surcharge_kwh_threshold']
    surcharge_rate = TOU_RATES_DATA['surcharge_rate_per_kwh']
    monthly_tou['surcharge'] = monthly_tou['kwh'].apply(lambda x: max(0, x - threshold) * surcharge_rate)
    monthly_tou['total_cost'] = monthly_tou['flow_cost'] + monthly_tou['basic_fee'] + monthly_tou['surcharge']
    total_cost_tou = monthly_tou['total_cost'].sum()
    
    monthly_prog = df_analysis.resample('MS').agg(kwh=('kwh', 'sum'))
    monthly_prog['is_summer'] = (monthly_prog.index.month >= 6) & (monthly_prog.index.month <= 9)
    monthly_prog['total_cost'] = monthly_prog.apply(lambda row: calculate_progressive_cost(row['kwh'], row['is_summer']), axis=1)
    total_cost_progressive = monthly_prog['total_cost'].sum()
    
    results = {'total_kwh': df_analysis['kwh'].sum(), 'cost_progressive': total_cost_progressive, 'cost_tou': total_cost_tou}
    return results, df_analysis

# --- 4. 核心 KPI 計算函式 (保持不變) ---
def get_core_kpis(df_history):
    kpis = {
        'projected_cost': 0, 'kwh_this_month_so_far': 0, 'kwh_last_7_days': 0,
        'kwh_previous_7_days': 0, 'weekly_delta_percent': 0, 'status_data_available': False,
        'peak_kwh': 0, 'off_peak_kwh': 0, 'PRICE_PER_KWH_AVG': 3.5,
        'kwh_today_so_far': 0, 'cost_today_so_far': 0, 'latest_data': None
    }
    
    if df_history.empty:
        return kpis

    try:
        kwh_last_30d = df_history.last('30D')['power_kW'].sum() * 0.25
        today = df_history.index.max()
        is_summer_now = (today.month >= 6) & (today.month <= 9)
        kpis['projected_cost'] = calculate_progressive_cost(kwh_last_30d, is_summer_now)
        if kwh_last_30d > 0:
            kpis['PRICE_PER_KWH_AVG'] = kpis['projected_cost'] / kwh_last_30d
        
        today_start = df_history.index.max().normalize()
        df_today = df_history.loc[today_start:]
        kpis['kwh_today_so_far'] = (df_today['power_kW'].sum() * 0.25)
        kpis['cost_today_so_far'] = kpis['kwh_today_so_far'] * kpis['PRICE_PER_KWH_AVG']

        today_date = df_history.index.max().date()
        start_of_month = today_date.replace(day=1)
        if start_of_month < df_history.index.min().date():
            start_of_month = df_history.index.min().date()
        df_this_month = df_history.loc[start_of_month:]
        kpis['kwh_this_month_so_far'] = (df_this_month['power_kW'].sum() * 0.25)

        df_last_7d = df_history.last('7D')
        kpis['kwh_last_7_days'] = (df_last_7d['power_kW'].sum() * 0.25)
        start_of_prev_7d = (df_last_7d.index.min() - timedelta(days=7))
        end_of_prev_7d = df_last_7d.index.min()
        
        if start_of_prev_7d >= df_history.index.min():
            df_prev_7d = df_history.loc[start_of_prev_7d:end_of_prev_7d]
            kpis['kwh_previous_7_days'] = (df_prev_7d['power_kW'].sum() * 0.25)
            if kpis['kwh_previous_7_days'] > 0: 
                kpis['weekly_delta_percent'] = ((kpis['kwh_last_7_days'] - kpis['kwh_previous_7_days']) / kpis['kwh_previous_7_days']) * 100
            kpis['status_data_available'] = True
        
        df_last_30d = df_history.last('30D').copy()
        tou_details_30d = df_last_30d.index.map(get_tou_details)
        df_last_30d['tou_category'] = [cat for cat, rate, season in tou_details_30d]
        df_last_30d['kwh'] = df_last_30d['power_kW'] * 0.25
        kpis['peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'peak']['kwh'].sum()
        kpis['off_peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'off_peak']['kwh'].sum()

        kpis['latest_data'] = df_history.iloc[-1]

        return kpis

    except Exception as e:
        return kpis
    
    
# ==========================================
# 🧪 測試區塊 (只在單獨執行此檔案時才會跑)
# ==========================================
if __name__ == "__main__":
    print("\nStarting data fetch test... (開始測試抓取雲端資料)")
    
    # 1. 測試抓取數據
    # 注意：因為沒有 Streamlit 環境，這裡呼叫 load_data 可能會因為 cache 警告而顯示訊息，這是正常的
    try:
        df_result = load_data()
        
        if df_result.empty:
            print("\n❌ 測試結果：抓取失敗，DataFrame 是空的。")
            print("可能原因：")
            print("1. Pantry ID 錯誤")
            print("2. 該年份的籃子 (Basket) 不存在 (例如 2023-q1)")
            print("3. 網路連線問題")
        else:
            print(f"\n✅ 測試成功！共抓取到 {len(df_result)} 筆資料。")
            print("\n數據預覽 (前 5 筆)：")
            print(df_result.head())
            print("\n數據預覽 (後 5 筆)：")
            print(df_result.tail())
            
            # 2. 測試 KPI 計算
            print("\n正在測試 KPI 計算...")
            kpis = get_core_kpis(df_result)
            print(f"本月累積用電: {kpis['kwh_this_month_so_far']:.2f} kWh")
            print(f"預估電費: ${kpis['projected_cost']:.0f}")

    except Exception as e:
        print(f"\n❌ 發生程式錯誤: {e}")