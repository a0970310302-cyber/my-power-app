# page_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np 

from app_utils import load_data, get_core_kpis

# --- 模擬帳單週期與費率計算函式 ---
def get_billing_status(current_kwh, predicted_kwh_add=0):
    start_date = "2026-01-01"
    end_date = "2026-03-01"
    
    # 簡易累進費率模擬
    if current_kwh <= 500:
        current_bill = current_kwh * 3.5
    else:
        current_bill = 500 * 3.5 + (current_kwh - 500) * 5.0
        
    # AI 預估結算 = 目前已知 + 未來預測總和
    # 如果有傳入 AI 預測值 (predicted_kwh_add)，就用 AI 的，否則用簡單估算
    if predicted_kwh_add > 0:
        # 這裡簡單假設未來每天都跟預測的 24 小時一樣 (粗略估算剩餘天數)
        # 實務上應該要有長期的預測，這裡先用短期預測 * 30 天做演示
        estimated_future_bill = predicted_kwh_add * 30 * 3.5 
        predicted_total_bill = current_bill + estimated_future_bill
    else:
        predicted_total_bill = current_bill * 1.8 
    
    budget_target = 3000 
    
    return {
        "period": f"{start_date} ~ {end_date}",
        "current_bill": int(current_bill),
        "predicted_bill": int(predicted_total_bill),
        "budget": budget_target
    }

def show_dashboard_page():
    """
    顯示「用電儀表板」的內容
    """
    # --- 1. 嘗試從 Session State 獲取最新的合併數據 ---
    if "current_data" in st.session_state and st.session_state.current_data is not None:
        df_history = st.session_state.current_data
        data_source_msg = "🟢 即時數據 (Live Data)"
    else:
        # Fallback 到讀取 CSV
        df_history = load_data()
        data_source_msg = "🟠 歷史存檔 (Offline Data)"
    
    if df_history is None or df_history.empty:
        st.warning("儀表板無資料可顯示。")
        return

    kpis = get_core_kpis(df_history)

    st.title("💡 家庭智慧電管家")
    st.caption(f"{data_source_msg} | AI 滾動修正模組：Online") 

    if not kpis['status_data_available']:
        st.warning("資料量不足，部分指標可能無法計算。")

    # ==========================================
    # 區塊 1: 帳單監控
    # ==========================================
    st.header("💰 帳單預算監控")
    
    # 計算未來 24 小時預測總量 (如果有)
    pred_sum_24h = 0
    if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
        pred_sum_24h = st.session_state.prediction_result['預測值'].sum()

    bill_status = get_billing_status(kpis['kwh_this_month_so_far'], predicted_kwh_add=pred_sum_24h)
    
    st.info(f"📅 **本期帳單週期： {bill_status['period']}**")
    
    c1, c2 = st.columns(2)
    c1.metric("💸 目前累積電費 (已知)", f"NT$ {bill_status['current_bill']:,}", delta="已定案")
    
    delta_val = bill_status['predicted_bill'] - bill_status['budget']
    delta_msg = f"超支 {delta_val} 元" if delta_val > 0 else f"省下 {abs(delta_val)} 元"
    delta_color = "inverse"
    
    c2.metric("🔮 AI 預估結算 (本期)", f"NT$ {bill_status['predicted_bill']:,}", 
              delta=delta_msg, delta_color=delta_color)

    usage_percent = min(bill_status['predicted_bill'] / bill_status['budget'], 1.0)
    st.write(f"**預算消耗進度 (目標：NT$ {bill_status['budget']:,})**")
    
    if usage_percent > 0.9:
        bar_caption = f"⚠️ 警告：預測即將超支！目前預測佔預算 {usage_percent*100:.1f}%"
    else:
        bar_caption = f"✅ 狀態良好：目前預測佔預算 {usage_percent*100:.1f}%"
    
    st.progress(usage_percent)
    st.caption(bar_caption)
    
    st.divider()

    # ==========================================
    # 區塊 2: 即時用電
    # ==========================================
    st.subheader("⚡ 即時用電狀態")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("今日累積用電", f"{kpis['kwh_today_so_far']:.2f} kWh")
    
    latest_data = kpis['latest_data']
    yesterday_power = 0
    instant_delta = 0
    
    try:
        yesterday_time = latest_data.name - timedelta(days=1)
        # 用 asof 找最接近的時間點比較保險
        if not df_history.empty:
            idx = df_history.index.get_indexer([yesterday_time], method='nearest')[0]
            yesterday_power = df_history.iloc[idx]['power_kW']
            if yesterday_power > 0:
                instant_delta = ((latest_data['power_kW'] - yesterday_power)/yesterday_power)*100
    except:
        pass
    
    k2.metric("當前功率", f"{latest_data['power_kW']:.3f} kW", f"{instant_delta:.1f}% vs 昨日")
    k3.metric("近 7 天累積", f"{kpis['kwh_last_7_days']:.1f} kWh")
    k4.metric("本期累積用量", f"{kpis['kwh_this_month_so_far']:.1f} kWh")

    st.divider()

    # ==========================================
    # 區塊 3: 滾動預測趨勢圖 (修正版：視覺截斷法)
    # ==========================================
    st.subheader("📈 雙月滾動式修正趨勢")
    
    tab1, tab2 = st.tabs(["預測 vs 真實", "詳細歷史數據"])
    
    with tab1:
        # 1. 準備歷史資料 (最近 3 天)
        # 【關鍵修改】過濾掉最後面是 0 或 NaN 的資料，避免圖表畫出「跳水」
        df_hist_plot = df_history.last('3D').copy()
        
        # 遞迴檢查：如果最後一筆是 0 或 NaN，就把它切掉，直到找到有值的
        # 這能製造出「斷開」的視覺效果，代表「這裡沒資料了」
        if not df_hist_plot.empty:
            while not df_hist_plot.empty and (df_hist_plot.iloc[-1]['power_kW'] <= 0 or pd.isna(df_hist_plot.iloc[-1]['power_kW'])):
                df_hist_plot = df_hist_plot.iloc[:-1]

        df_hist_plot = df_hist_plot[['power_kW']].reset_index()
        df_hist_plot.columns = ['time', 'value']
        df_hist_plot['type'] = '真實數據 (Actual)'
        
        # 2. 準備預測資料
        df_pred_plot = pd.DataFrame()
        if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
            pred_res = st.session_state.prediction_result.copy()
            
            # 【關鍵修改】讓預測線跟歷史線「無縫接軌」
            # 我們把歷史數據的最後一個點，加到預測數據的最前面，這樣圖表中間就不會斷掉
            if not df_hist_plot.empty:
                last_hist_point = pd.DataFrame({
                    'time': [df_hist_plot.iloc[-1]['time']], 
                    'value': [df_hist_plot.iloc[-1]['value']],
                    'type': ['AI 預測 (Forecast)'] # 標記為預測，讓顏色跟後面一致
                })
                # 預測值本身
                future_pred = pred_res[['預測值']].reset_index()
                future_pred.columns = ['time', 'value']
                future_pred['type'] = 'AI 預測 (Forecast)'
                
                df_pred_plot = pd.concat([last_hist_point, future_pred])
            else:
                # 萬一真的沒歷史資料，直接畫預測
                df_pred_plot = pred_res[['預測值']].reset_index()
                df_pred_plot.columns = ['time', 'value']
                df_pred_plot['type'] = 'AI 預測 (Forecast)'

        # 合併並畫圖
        if not df_pred_plot.empty:
            df_chart = pd.concat([df_hist_plot, df_pred_plot])
            
            # 取得最後一個「真實」時間點，作為 Now 的標記
            last_real_time = df_hist_plot['time'].iloc[-1] if not df_hist_plot.empty else datetime.now()

            fig = px.line(df_chart, x='time', y='value', color='type', 
                          color_discrete_map={'真實數據 (Actual)': '#00CC96', 'AI 預測 (Forecast)': '#EF553B'},
                          line_dash='type',
                          line_dash_map={'真實數據 (Actual)': 'solid', 'AI 預測 (Forecast)': 'dash'},
                          title=f"負載預測 (最後更新: {last_real_time.strftime('%H:%M')})",
                          template="plotly_dark")
            
            # 標示 "Data Lag" 的界線
            fig.add_vline(x=last_real_time.timestamp() * 1000, line_width=1, line_dash="dot", line_color="white")
            fig.add_annotation(x=last_real_time.timestamp() * 1000, y=df_chart['value'].max(), 
                               text="即時訊號截止", showarrow=True, arrowhead=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示一個小小的提示，解釋為什麼會有虛線
            if (datetime.now() - last_real_time).total_seconds() > 3600:
                 st.info(f"ℹ️ 系統備註：監測到感測器訊號延遲。目前 **{last_real_time.strftime('%H:%M')}** 之後的數據由 AI 預測模型即時填補。")
        else:
            st.info("無法顯示預測圖表。")
        
        with st.expander("ℹ️ 技術原理：Hybrid Model"):
            st.write("""
            本系統結合 **LightGBM (擅長捕捉規律)** 與 **LSTM (擅長捕捉時序特徵)**。
            上方橘色虛線即為兩種模型加權後的最終預測結果。
            """)

    with tab2:
        st.dataframe(df_history.tail(100))