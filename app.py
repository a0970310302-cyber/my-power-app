import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px # 匯入 Plotly
import plotly.graph_objects as go # 匯入 Plotly 的 Graph Objects
import numpy as np # 匯入 Numpy
import json # ⭐ 為了 Lottie 動畫
from streamlit_lottie import st_lottie # ⭐ 為了 Lottie 動畫
import time # ⭐ 為了 Lottie 動畫

# --- 0. 頁面設定 (必須是第一個 st 指令) ---
st.set_page_config(layout="wide")

# 確保 data_loader.py 和 model_trainer.py 在同一個資料夾
try:
    from data_loader import load_all_history_data
    from model_trainer import create_features
except ImportError:
    st.error("錯誤：找不到 data_loader.py 或 model_trainer.py。請確保檔案位於專案根目錄中。")
    st.stop()


# --- 1. Lottie 動畫載入函式 ---
@st.cache_data
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"錯誤：找不到 Lottie 動畫檔案 '{filepath}'。")
        return None
    except Exception as e:
        st.error(f"載入本地 Lottie 檔案時發生錯誤：{e}")
        return None

# --- 2. 核心快取功能 (Caching) ---
@st.cache_resource
def load_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        st.error(f"錯誤：找不到模型檔案 '{model_path}'。請先執行 model_trainer.py 來產生模型。")
        return None
    try:
        time.sleep(2) # 模擬模型載入
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"載入模型時發生錯誤：{e}")
        return None

@st.cache_data
def load_data():
    try:
        time.sleep(1) # 模擬數據載入
        df_history = load_all_history_data()
        if df_history.empty:
            st.warning("警告：未載入任何歷史資料。請檢查您的 JSON 檔案。")
            return pd.DataFrame()
        return df_history
    except Exception as e:
        st.error(f"載入歷史資料時發生錯誤：{e}")
        return pd.DataFrame()

# --- 3. 電價計算邏輯 (共用) ---
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
    df_analysis['is_summer'] = [season for cat, rate, season in tou_details]
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

# --- 4. 核心 KPI 計算函式 ---
def get_core_kpis(df_history):
    """
    計算所有頁面共用的核心 KPI
    """
    # 初始化
    kpis = {
        'projected_cost': 0, 'kwh_this_month_so_far': 0, 'kwh_last_7_days': 0,
        'kwh_previous_7_days': 0, 'weekly_delta_percent': 0, 'status_data_available': False,
        'peak_kwh': 0, 'off_peak_kwh': 0, 'PRICE_PER_KWH_AVG': 3.5,
        'kwh_today_so_far': 0, 'cost_today_so_far': 0, 'latest_data': None
    }
    
    if df_history.empty:
        return kpis # 返回初始值

    try:
        # --- 預估電費 (累進) ---
        kwh_last_30d = df_history.last('30D')['power_kW'].sum() * 0.25
        today = df_history.index.max()
        is_summer_now = (today.month >= 6) & (today.month <= 9)
        kpis['projected_cost'] = calculate_progressive_cost(kwh_last_30d, is_summer_now)
        if kwh_last_30d > 0:
            kpis['PRICE_PER_KWH_AVG'] = kpis['projected_cost'] / kwh_last_30d
        
        # --- 今日數據 ---
        today_start = df_history.index.max().normalize()
        df_today = df_history.loc[today_start:]
        kpis['kwh_today_so_far'] = (df_today['power_kW'].sum() * 0.25)
        kpis['cost_today_so_far'] = kpis['kwh_today_so_far'] * kpis['PRICE_PER_KWH_AVG']

        # --- 本月累積 ---
        today_date = df_history.index.max().date()
        start_of_month = today_date.replace(day=1)
        if start_of_month < df_history.index.min().date():
            start_of_month = df_history.index.min().date()
        df_this_month = df_history.loc[start_of_month:]
        kpis['kwh_this_month_so_far'] = (df_this_month['power_kW'].sum() * 0.25)

        # --- 用電狀態 (週) ---
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
        
        # --- 尖峰/離峰 (TOU) ---
        df_last_30d = df_history.last('30D').copy()
        tou_details_30d = df_last_30d.index.map(get_tou_details)
        df_last_30d['tou_category'] = [cat for cat, rate, season in tou_details_30d]
        df_last_30d['kwh'] = df_last_30d['power_kW'] * 0.25
        kpis['peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'peak']['kwh'].sum()
        kpis['off_peak_kwh'] = df_last_30d[df_last_30d['tou_category'] == 'off_peak']['kwh'].sum()

        # --- 最新數據 ---
        kpis['latest_data'] = df_history.iloc[-1]

        return kpis

    except Exception as e:
        st.error(f"核心 KPI 計算錯誤: {e}")
        return kpis # 返回初始值


# --- 5. 頁面內容函式 ---

def show_home_page():
    """
    顯示新的「主頁」總覽
    """
    st.title("💡 智慧電能管家總覽")
    
    # --- 載入數據並計算 KPI ---
    df_history = load_data()
    kpis = get_core_kpis(df_history)

    if not kpis['status_data_available']:
        st.warning("??r(･x･｡)??? 歷史資料不足 (需 14 天) 或載入失敗，無法顯示總覽。")
        st.info("請檢查您的數據檔案。")
    else:
        # --- 顯示核心 KPI ---
        st.markdown("### 關鍵資訊總覽")
        
        # 1. 用電狀態
        if kpis['weekly_delta_percent'] > 10: status_display = f":red[(｡ ́︿ ̀｡) 警示]"
        elif kpis['weekly_delta_percent'] < -10: status_display = ":green[(๑•̀ㅂ•́)و✧ 良好]"
        else: status_display = ":blue[(・-・) 普通]"
        st.subheader(f"您本週的用電狀態： {status_display}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("本週累積用電 (近 7 天)", f"{kpis['kwh_last_7_days']:.2f} kWh")
        col2.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
        col3.metric("本月累積用電 (至今)", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()

    # --- 節能目標設定移至此處 ---
    st.markdown("### 💰 預算與目標")
    if 'cost_target' not in st.session_state:
        st.session_state.cost_target = 1000 
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.metric(
            label="預估本月總電費",
            value=f"{kpis['projected_cost']:.0f} 元",
            help="依據您過去30天的用電模式，以累進電價估算。"
        )
        current_target = st.session_state.get('cost_target', 1000)
        current_remaining = current_target - kpis['projected_cost']
        current_delta_color = "inverse" if current_remaining < 0 else "normal"
        st.metric(
            label="本月剩餘預算",
            value=f"{current_remaining:.0f} 元",
            delta_color=current_delta_color
        )
    with b_col2:
        st.session_state.cost_target = st.number_input(
            "請輸入您的本月電費目標 (元)",
            min_value=0,
            value=st.session_state.get('cost_target', 1000),
            step=100,
            key="cost_target_input"
        )
        
    st.divider()

    # --- 顯示功能說明 ---
    st.markdown("### 功能導覽")
    
    st.subheader("📈 用電儀表板")
    st.markdown("查看詳細的用電數據，包含：")
    st.markdown("- **即時用電** 與昨日同期比較\n- **最近 7 天** 的詳細用電曲線\n- **近 30 天** 的尖峰/離峰用電圓餅圖\n- **每日歷史數據** 的長條圖與資料")
    
    st.subheader("🔬 AI 決策分析室")
    st.markdown("利用 AI 模型進行深度分析：")
    st.markdown("- **AI 用電預測**：預測未來任一天的 15 分鐘用電曲線。\n- **AI 電價分析器**：回測歷史數據，比較「累進電價」與「時間電價」的成本，找出最適合您的方案。\n- **AI 用電異常分析**：自動偵測歷史數據中用電量異常飆高的時段。\n- **AI 節能建議**：根據您的電費目標，提供客製化節能建議。")


def show_dashboard_page():
    """
    顯示「用電儀表板」的內容
    """
    # --- 載入數據並計算 KPI ---
    df_history = load_data()
    kpis = get_core_kpis(df_history)

    # --- 儀表板頁面內容 ---
    st.title("💡 智慧電能管家")
    st.header("📈 用電儀表板")

    if df_history.empty or not kpis['status_data_available']:
        st.warning("儀表板無資料可顯示，或歷史資料不足 14 天。")
    else:
        # --- 本週用電狀態 ---
        if kpis['weekly_delta_percent'] > 10: status_display = f":red[(｡ ́︿ ̀｡) 警示]"
        elif kpis['weekly_delta_percent'] < -10: status_display = ":green[(๑•̀ㅂ•́)و✧ 良好]"
        else: status_display = ":blue[(・-・) 普通]"
        st.subheader(f"您的用電狀態： {status_display}")
        
        # --- KPI 控制中心 ---
        st.markdown("### 關鍵指標 (KPI) 控制中心")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
        col2.metric("今日預估電費", f"{kpis['cost_today_so_far']:.0f} 元")
        col3.metric("本週累積用電 (近 7 天)", f"{kpis['kwh_last_7_days']:.2f} kWh")
        col4.metric("本月累積用電 (至今)", f"{kpis['kwh_this_month_so_far']:.1f} kWh")
        
        col5, col6 = st.columns(2)
        latest_data = kpis['latest_data']
        latest_power = latest_data['power_kW']
        yesterday_time = latest_data.name - timedelta(days=1)
        instant_delta_text, instant_delta_color, yesterday_power_display = "N/A", "off", "N/A"
        
        if yesterday_time in df_history.index:
            yesterday_data = df_history.loc[yesterday_time]
            yesterday_power = yesterday_data['power_kW']
            yesterday_power_display = f"{yesterday_power:.3f} kW"
            if yesterday_power > 0:
                instant_delta = ((latest_power - yesterday_power) / yesterday_power) * 100
                if instant_delta > 10: instant_delta_text = f"高於昨日 {instant_delta:.1f}%"; instant_delta_color = "inverse"
                elif instant_delta < -10: instant_delta_text = f"低於昨日 {abs(instant_delta):.1f}%"; instant_delta_color = "normal"
                else: instant_delta_text = f"{instant_delta:+.1f}%"; instant_delta_color = "normal"
            else: instant_delta_text = "昨日無耗電"
        else: instant_delta_text = "無昨日資料"
        
        col5.metric(label=f"最新用電功率 ({latest_data.name.strftime('%H:%M')})", value=f"{latest_power:.3f} kW")
        col6.metric(label=f"昨日同期 ({yesterday_time.strftime('%H:%M')})", value=yesterday_power_display, delta=instant_delta_text, delta_color=instant_delta_color)
        
        st.divider() 

        # --- 圖表 Tabs ---
        st.subheader("用電趨勢分析")
        tab1, tab2, tab3 = st.tabs(["📈 最近 7 天趨勢", "🍩 近 30 天尖離峰", "📊 每日歷史數據"])

        with tab1:
            st.markdown("##### 最近 7 天用電曲線")
            df_7d = df_history.last('7D')['power_kW'].reset_index()
            df_7d.columns = ['時間', '功率 (kW)']
            fig_line = px.line(df_7d, x='時間', y='功率 (kW)', template="plotly_dark")
            fig_line.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)
            st.plotly_chart(fig_line, use_container_width=True)
            with st.expander("📖 顯示最近 7 天的 15 分鐘原始數據"):
                st.dataframe(df_7d.set_index('時間'))

        with tab2:
            st.markdown("##### 近 30 天尖離峰佔比 (TOU)")
            if kpis['peak_kwh'] + kpis['off_peak_kwh'] > 0:
                labels = ['尖峰用電', '離峰用電']
                # 【修正筆誤】kpis['off_kwh_tou'] 應為 kpis['off_peak_kwh']
                values = [kpis['peak_kwh'], kpis['off_peak_kwh']] 
                colors = ['#FF6B6B', '#4ECDC4'] 
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=.4, 
                    marker=dict(colors=colors, line=dict(color='#333', width=1))
                )])
                fig_donut.update_layout(
                    template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                st.info("此圖表是基於「簡易型時間電價 (TOU)」的時段定義來劃分您的用電分佈。")
            else:
                st.info("無足夠資料可分析尖離峰佔比。")
                
        with tab3:
            st.markdown("##### 每日用電量 (kWh) 長條圖")
            df_daily_kwh = (df_history['power_kW'].resample('D').sum() * 0.25).to_frame(name="每日總度數 (kWh)")
            min_date = df_daily_kwh.index.min().date()
            max_date = df_daily_kwh.index.max().date()
            default_start_date = max(min_date, max_date - timedelta(days=30))
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("選擇日期範圍 - 開始", value=default_start_date, min_value=min_date, max_value=max_date, key="hist_start")
            with col_date2:
                end_date = st.date_input("選擇日期範圍 - 結束", value=max_date, min_value=start_date, max_value=max_date, key="hist_end")
            filtered_daily_df = df_daily_kwh.loc[start_date:end_date]
            st.markdown(f"**{start_date} 至 {end_date} 數據**")
            fig_bar = px.bar(filtered_daily_df, y='每日總度數 (kWh)', template="plotly_dark")
            fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            with st.expander("📖 顯示每日數據表格"):
                st.dataframe(filtered_daily_df.style.format("{:.2f}"))

def show_analysis_page():
    """
    顯示「AI 決策分析室」的內容
    """
    # --- 載入數據並計算 KPI (為了 Tab 4) ---
    model = load_model()
    df_history = load_data()
    kpis = get_core_kpis(df_history)

    # --- AI 決策分析室頁面內容 ---
    st.header("🔬 AI 決策分析室")
    st.info("利用 AI 模型預測未來用電，並分析您的最佳電價方案。")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI 用電預測",  
        "💰 AI 電價分析器",
        "⚠️ AI 用電異常分析",
        "🎯 AI 節能建議"
        ])

    # --- AI 預測分頁 ---
    with tab1:
        st.subheader("🤖 AI 用電預測")
        
        if model is None or df_history.empty:
            st.error("模型或歷史資料載入失敗，無法進行預測。")
        else:
            default_future_date = df_history.index.max().date() + timedelta(days=1)
            future_date = st.date_input(
                "請選擇您要預測的日期：",
                value=default_future_date,
                min_value=df_history.index.min().date() + timedelta(days=1),
                max_value=df_history.index.max().date() + timedelta(days=30),
                help="AI 將根據歷史數據，預測您所選日期當天的 15 分鐘用電曲線。"
            )

            if st.button("📈 開始預測"):
                with st.spinner("AI 正在為您計算... (這可能需要幾秒鐘)"):
                    try:
                        future_timestamps = pd.date_range(start=future_date, periods=96, freq='15T')
                        df_future = pd.DataFrame(index=future_timestamps)
                        
                        lag_date = future_date - timedelta(days=1)
                        lag_data_time = future_timestamps - timedelta(days=1)
                        
                        try:
                            lag_df = df_history.loc[lag_data_time]
                            lag_df = lag_df.set_index(future_timestamps)
                            df_future['lag_1_day'] = lag_df['power_kW']
                        except KeyError:
                            st.error(f"錯誤：找不到 {lag_date.strftime('%Y-%m-%d')} 的完整歷史資料，無法產生『昨日同期』特徵。")
                            df_future['lag_1_day'] = 0  
                            st.warning("已使用 0 填充 'lag_1_day' 特徵。")
                        except Exception as e:
                            st.error(f"提取 Lag 特徵時發生未知錯誤：{e}")
                            raise  

                        df_future_with_feats = create_features(df_future)
                        FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'is_weekend', 'lag_1_day']
                        
                        missing_features = [f for f in FEATURES if f not in df_future_with_feats.columns]
                        if missing_features:
                            raise ValueError(f"即時特徵工程中缺少以下特徵：{missing_features}")

                        X_future = df_future_with_feats[FEATURES]
                        prediction = model.predict(X_future)
                        df_pred = pd.DataFrame(prediction, index=future_timestamps, columns=['預測用電 (kW)'])
                        
                        st.subheader(f"📅 {future_date.strftime('%Y-%m-%d')} 預測結果")
                        
                        total_kwh = df_pred['預測用電 (kW)'].sum() * 0.25  
                        peak_power = df_pred['預測用電 (kW)'].max()
                        peak_time = df_pred['預測用電 (kW)'].idxmax().strftime('%H:%M')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("預測總度數 (kWh)", f"{total_kwh:.2f} 度")
                        with col2:
                            st.metric("預測用電高峰", f"{peak_power:.3f} kW", f"發生在 {peak_time}")
                        
                        fig_pred = px.line(df_pred, y='預測用電 (kW)', template="plotly_dark", color_discrete_sequence=['#FF6B6B'])
                        fig_pred.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with st.expander("📖 顯示預測的 15 分鐘原始數據"):
                            st.dataframe(df_pred.style.format("{:.3f} kW"))
                    
                    except ValueError as ve:
                        st.error(f"執行 AI 預測時發生錯誤：{ve}")
                    except Exception as e:
                        st.error(f"執行 AI 預測時發生未知錯誤：{e}")

    # --- AI 電價分析器分頁 ---
    with tab2:
        st.subheader("💰 AI 電價分析器 (依據2024/4/1電價)")
        
        if df_history.empty:
            st.warning("無歷史資料可供分析。")
        else:
            st.markdown("此功能將回測您的歷史用電數據，比較 **「累進電價」** 與 **「簡易型時間電價 (TOU)」** 的總成本。")
            
            with st.expander("點此查看電價方案詳情"):
                st.markdown("##### 方案一：累進電價 (一般住宅預設)")
                st.markdown("""
                | 每月用電度數 (kWh) | 夏月 (6-9月) | 非夏月 |
                | :--- | :---: | :---: |
                | 120 度以下 | 1.68 元 | 1.68 元 |
                | 121~330 度 | 2.45 元 | 2.16 元 |
                | 331~500 度 | 3.70 元 | 3.03 元 |
                | 501~700 度 | 5.04 元 | 4.14 元 |
                | 701~1000 度 | 6.24 元 | 5.07 元 |
                | 1001 度以上 | 8.46 元 | 6.63 元 |
                """)
                
                st.markdown("##### 方案二：簡易型時間電價 (TOU) - 二段式")
                st.markdown(f"- **基本電費：** 每月 `{TOU_RATES_DATA['basic_fee_monthly']}` 元")
                st.markdown(f"- **夏月 (6/1-9/30)**")
                st.markdown(f"  - **尖峰 (週一至五 09:00-24:00)：** `{TOU_RATES_DATA['rates']['summer']['peak']}` 元/度")
                st.markdown(f"  - **離峰 (尖峰以外 + 假日)：** `{TOU_RATES_DATA['rates']['summer']['off_peak']}` 元/度")
                st.markdown(f"- **非夏月**")
                st.markdown(f"  - **尖峰 (週一至五 06:00-11:00, 14:00-24:00)：** `{TOU_RATES_DATA['rates']['nonsummer']['peak']}` 元/度")
                st.markdown(f"  - **離峰 (尖峰以外 + 假日)：** `{TOU_RATES_DATA['rates']['nonsummer']['off_peak']}` 元/度")
                st.markdown(f"*注意：每月總用電量超過 {TOU_RATES_DATA['surcharge_kwh_threshold']} 度，超過部分每度加收 {TOU_RATES_DATA['surcharge_rate_per_kwh']} 元。*")

            st.markdown("---")
            st.markdown("##### 選擇您要分析的歷史資料範圍")
            min_date = df_history.index.min().date()
            max_date = df_history.index.max().date()
            default_start_date = max(min_date, max_date - timedelta(days=29))  

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("分析開始日期", value=default_start_date, min_value=min_date, max_value=max_date, key="analysis_start")
            with col_date2:
                end_date = st.date_input("分析結束日期", value=max_date, min_value=start_date, max_value=max_date, key="analysis_end")
            
            analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].copy()

            if st.button("💰 開始分析電價"):
                if analysis_df.empty:
                    st.error("選定範圍內無資料，請重新選擇日期。")
                else:
                    with st.spinner("AI 正在回測您的歷史用電..."):
                        try:
                            results, df_detailed = analyze_pricing_plans(analysis_df)
                            
                            cost_prog = results['cost_progressive']
                            cost_tou = results['cost_tou']
                            total_kwh = results['total_kwh']

                            st.subheader(f"📅 {start_date} 至 {end_date} 電價分析結果")
                            st.markdown(f"期間總用電量： **{total_kwh:,.2f} kWh**")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("方案一：累進電價 (標準)", f"{cost_prog:,.0f} 元")
                            col2.metric("方案二：簡易型時間電價 (TOU)", f"{cost_tou:,.0f} 元")
                            
                            st.divider()
                            
                            difference = cost_prog - cost_tou
                            if difference > 0:
                                best_plan = "簡易型時間電價 (TOU)"
                                savings = difference
                                st.success(f"**分析建議：:green[(๑•̀ㅂ•́)و✧]**")
                                st.success(f"在此期間，若選用 **{best_plan}**，預計可**節省 {savings:,.0f} 元**！")
                                st.info("您的用電模式可能在離峰時段佔比較高。")
                            else:
                                best_plan = "累進電價 (標準)"
                                savings = abs(difference)
                                st.warning(f"**分析建議：:red[(｡ ́︿ ̀｡)]**")
                                st.warning(f"在此期間，選用 **{best_plan}** 較為划算 (可省 {savings:,.0f} 元)。")
                                st.info(f"若要改用時間電價，建議您將尖峰用電轉移至離峰時段。")
                                
                            st.markdown("---")
                            st.subheader("TOU 用電分佈 (kWh)")
                            
                            df_kwh_dist = df_detailed.groupby('tou_category')['kwh'].sum().reset_index()
                            
                            fig_pie_kwh = px.pie(df_kwh_dist, names='tou_category', values='kwh', 
                                                 title='TOU 時段用電量 (kWh) 分佈',
                                                 color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#4ECDC4'},
                                                 template="plotly_dark")
                            st.plotly_chart(fig_pie_kwh, use_container_width=True)
                            
                            st.subheader("TOU 成本分佈 (時間電價)")
                            df_cost_dist = df_detailed.groupby('tou_category')['tou_flow_cost'].sum().reset_index()
                            
                            fig_pie_cost = px.pie(df_cost_dist, names='tou_category', values='tou_flow_cost', 
                                                  title='TOU 時段電費 (元) 分佈',
                                                  color_discrete_map={'peak':'#FF6B6B', 'off_peak':'#4ECDC4'},
                                                  template="plotly_dark")
                            st.plotly_chart(fig_pie_cost, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"執行電價分析時發生錯誤: {e}")
                            st.error("請檢查您的資料範圍是否完整。")

    # --- 異常分析分頁 ---
    with tab3:
        st.subheader("⚠️ AI 用電異常分析")
        
        if df_history.empty:
            st.warning("無歷史資料可供分析。")
        else:
            st.markdown("此功能將分析您的完整歷史數據，找出用電量顯著高於平時的時段。")
            
            with st.spinner("AI 正在分析您的歷史數據..."):
                try:
                    df_analysis_anomaly = df_history.copy()
                    window_size = 96 * 7
                    df_analysis_anomaly['rolling_avg'] = df_analysis_anomaly['power_kW'].rolling(window=window_size, center=True, min_periods=96).mean()
                    df_analysis_anomaly['rolling_std'] = df_analysis_anomaly['power_kW'].rolling(window=window_size, center=True, min_periods=96).std()
                    df_analysis_anomaly['anomaly_threshold'] = df_analysis_anomaly['rolling_avg'] + (2 * df_analysis_anomaly['rolling_std'])
                    
                    anomalies = df_analysis_anomaly[df_analysis_anomaly['power_kW'] > df_analysis_anomaly['anomaly_threshold']]

                    if anomalies.empty:
                        st.success("🎉 分析完畢：在您的歷史數據中未發現明顯的用電異常事件。")
                    else:
                        st.warning(f"偵測到 {len(anomalies)} 筆 (15分鐘) 異常用電事件！")
                        st.markdown("---")
                        st.markdown("#### 異常用電時段 vs 歷史平均 (最近 30 天)")
                        
                        chart_data = df_analysis_anomaly.last('30D')[[
                            'power_kW', 'rolling_avg', 'anomaly_threshold'
                        ]]
                        chart_data.columns = ['實際用電', '7日平均', '異常閾值']
                        
                        fig_anomaly = px.line(chart_data, template="plotly_dark")
                        fig_anomaly.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### 異常事件詳細列表")
                        
                        with st.expander("📖 顯示異常事件的 15 分鐘原始數據"):
                            st.dataframe(anomalies[['power_kW', 'rolling_avg', 'anomaly_threshold']])

                except Exception as e:
                    st.error(f"執行異常分析時發生錯誤：{e}")

    # --- AI 節能建議分頁 ---
    with tab4:
        st.subheader("🎯 AI 節能建議")
        
        # 這裡的 'cost_target' 會從側邊欄的 st.session_state 讀取
        target_cost = st.session_state.get('cost_target', 1000) 
        st.info(f"您在主頁設定的本月電費目標為： **{target_cost} 元**")
        
        if df_history.empty:
            st.warning("無歷史資料，無法進行節能建議。")
        else:
            with st.spinner("AI 正在分析您的節能潛力..."):
                try:
                    # 'projected_cost' 和 'PRICE_PER_KWH_AVG' 
                    # 是在這個函式開頭的 "get_core_kpis" 函式計算的
                    difference = kpis['projected_cost'] - target_cost
                    st.markdown("---")
                    
                    if difference > 0:
                        st.error(f"**警示：:red[(｡ ́︿ ̀｡)]**")
                        st.error(f"以您過去 30 天的用電模式估算，本月電費約為 **{kpis['projected_cost']:.0f} 元** (依累進電價計算)，將**超過**您的目標 **{difference:.0f} 元**。")
                        
                        st.markdown("#### 💡 AI 節能建議：")
                        daily_kwh_reduction_needed = (difference / kpis['PRICE_PER_KWH_AVG']) / 30
                        st.markdown(f"* 您需要**每日平均減少 {daily_kwh_reduction_needed:.2f} 度 (kWh)** 的用電量才能達標。")
                        st.markdown(f"* **建議您：**")
                        st.markdown(f"    1.  前往「**AI 電價分析器**」分頁，確認您是否使用了最划算的電價方案。")
                        st.markdown(f"    2.  前往「**AI 用電異常分析**」分頁，找出您的異常高耗電時段。")
                        
                    else:
                        st.success(f"**恭喜！:green[(๑•̀ㅂ•́)و✧]**")
                        st.success(f"以您過去 30 天的用電模式估算，本月電費約為 **{kpis['projected_cost']:.0f} 元** (依累進電價計算)，**低於**您的 **{target_cost} 元** 目標。")
                        st.markdown("#### 💡 AI 節能建議：")
                        st.markdown("* 您的用電習慣非常良好！")
                        st.markdown("* 可以前往「**AI 電價分析器**」分頁，看看是否有機會省下更多錢！")

                except Exception as e:
                    st.error(f"執行節能建議分析時發生錯誤：{e}")


# --- 6. 主程式：開場動畫 ---
if "app_ready" not in st.session_state:
    st.session_state.app_ready = False

if not st.session_state.app_ready:
    lottie_filepath = "loading_animation.json" # 確保此檔案在 app.py 旁邊
    lottie_json = load_lottiefile(lottie_filepath)
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_json:
            st_lottie(lottie_json, speed=1, width=400, height=400, key="loading_lottie")
        else:
            st.warning("動畫載入失敗...")
        
        st.subheader("💡 智慧電能管家 啟動中...")
        st.text("正在為您載入 AI 模型與歷史數據...")

    # 觸發快取函式
    model = load_model()
    df_history = load_data()

    if model is not None and not df_history.empty and lottie_json is not None:
        st.session_state.app_ready = True
        st.session_state.page = "🏠 主頁" # 預設頁面
        st.rerun()
    else:
        st.error("啟動失敗：無法載入模型、數據或動畫。請檢查您的檔案。")
        st.stop()

# --- 7. 主程式：手動側邊欄 與 頁面路由 ---
# (只有在 app_ready = True 時才會執行)

with st.sidebar:
    # 1. 【⭐ 修改點】將 Logo 圖片替換為 cat.json Lottie 動畫
    lottie_cat = load_lottiefile("idn.json") # 載入 cat.json
    if lottie_cat:
        st_lottie(
            lottie_cat,
            speed=1,
            loop=True,  # 確保動畫循環播放
            quality="high", 
            height=150,     # 您可以調整適合的高度
            key="cat_animation"
        )
    else:
        # 如果 cat.json 載入失敗，顯示備用文字
        st.header("AI Power Forecast")
        st.warning("cat.json 動畫載入失敗")
        
    # 2. 再放標題
    st.header("功能選單")
    st.divider()

    # 3. 自定義導覽按鈕
    if st.button("🏠 主頁", key="nav_home", use_container_width=True, type="secondary" if st.session_state.get('page', '🏠 主頁') != "🏠 主頁" else "primary"):
        st.session_state.page = "🏠 主頁"
        st.rerun() # 點擊按鈕時強制刷新
    
    if st.button("📈 用電儀表板", key="nav_dashboard", use_container_width=True, type="secondary" if st.session_state.get('page') != "📈 用電儀表板" else "primary"):
        st.session_state.page = "📈 用電儀表板"
        st.rerun()

    if st.button("🔬 AI 決策分析室", key="nav_analysis", use_container_width=True, type="secondary" if st.session_state.get('page') != "🔬 AI 決策分析室" else "primary"):
        st.session_state.page = "🔬 AI 決策分析室"
        st.rerun()

    # --- 【⭐ 修改點：節能目標設定已從此處移除】 ---
    # st.divider()
    # st.header("🎯 節能目標設定")
    # ... (相關程式碼已被移至 show_home_page() 函式中) ...

# --- 頁面路由 ---
# 根據 st.session_state.page 的值來顯示對應的函式
if st.session_state.get('page') == "📈 用電儀表板":
    show_dashboard_page()
elif st.session_state.get('page') == "🔬 AI 決策分析室":
    show_analysis_page()
else: # 預設或 "🏠 主頁"
    show_home_page()