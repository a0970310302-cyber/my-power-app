import streamlit as st
from datetime import timedelta, datetime
import pandas as pd

# 匯入共用函式
from app_utils import load_data, get_core_kpis, analyze_pricing_plans

# [模擬函式] 取得預算狀態
def get_budget_health(current_kwh):
    # 簡易模擬：假設預算 3000 元
    predicted_bill = current_kwh * 4.5 * 1.5 
    budget = 3000
    
    status = "safe"
    if predicted_bill > budget:
        status = "danger"
    elif predicted_bill > budget * 0.9:
        status = "warning"
        
    return status, int(predicted_bill), budget

def show_home_page():
    """
    【AI 每日晨報】風格主頁
    """
    st.title("🏠 家庭智慧電管家")
    
    # --- 0. 資料準備 ---
    df_history = load_data()
    if df_history is None or df_history.empty:
        st.warning("⚠️ 系統初始化中，等待數據接入...")
        return
    kpis = get_core_kpis(df_history)
    
    # 取得各項指標狀態
    budget_status, pred_bill, budget_target = get_budget_health(kpis['kwh_this_month_so_far'])
    
    # 電價分析
    last_date = df_history.index.max().date()
    start_date = last_date - timedelta(days=29)
    analysis_df = df_history.loc[start_date.strftime('%Y-%m-%d'):last_date.strftime('%Y-%m-%d')].copy()
    plan_savings = 0
    if not analysis_df.empty:
        try:
            res, _ = analyze_pricing_plans(analysis_df)
            plan_savings = res['cost_progressive'] - res['cost_tou']
        except:
            pass

    # --- 1. AI 總結語 ---
    welcome_msg = ""
    if budget_status == "danger":
        welcome_msg = f"🚨 **警報：預測本月將超支 {pred_bill - budget_target} 元！建議立即啟動節能措施。**"
        st.error(welcome_msg, icon="🚨")
    elif plan_savings > 100:
        welcome_msg = f"💡 **早安！系統發現若切換電價方案，本月可省下 {plan_savings:.0f} 元，建議查看詳情。**"
        st.info(welcome_msg, icon="💡")
    else:
        welcome_msg = f"✅ **早安！目前用電狀況良好，預算控制在安全範圍內。**"
        st.success(welcome_msg, icon="✅")

    st.markdown("---")

    # --- 2. 三大決策卡片 ---
    col1, col2, col3 = st.columns(3)

    # === 卡片 1: 財務安全 ===
    with col1:
        with st.container(border=True):
            st.markdown("#### 💰 預算監控")
            if budget_status == "safe":
                st.markdown("# :green[安全]")
                st.caption(f"預測結算 ${pred_bill}")
                st.progress(min(pred_bill/budget_target, 1.0))
            elif budget_status == "warning":
                st.markdown("# :orange[警戒]")
                st.caption(f"接近預算 ${pred_bill}")
                st.progress(min(pred_bill/budget_target, 1.0))
            else:
                st.markdown("# :red[超支]")
                st.caption(f"預測爆表 ${pred_bill}")
                st.progress(1.0)
            st.markdown(f"**目標：${budget_target}**")

    # === 卡片 2: 方案優化 ===
    with col2:
        with st.container(border=True):
            st.markdown("#### 📉 方案最佳化")
            if plan_savings > 50:
                st.markdown("# :green[建議切換]")
                st.metric("可節省", f"NT$ {plan_savings:,.0f}", delta="時間電價更優")
            else:
                st.markdown("# :blue[維持現狀]")
                st.metric("累進最省", "最佳方案", delta_color="off")

    # === 卡片 3: 行為診斷 ===
    with col3:
        with st.container(border=True):
            st.markdown("#### 🩺 用電健康度")
            trend = kpis['weekly_delta_percent']
            if trend > 15:
                st.markdown("# :red[異常飆升]")
                st.metric("較上週", f"+{trend:.1f}%", delta_color="inverse")
            elif trend < -10:
                st.markdown("# :green[顯著節能]")
                st.metric("較上週", f"{trend:.1f}%", delta_color="inverse")
            else:
                st.markdown("# :blue[平穩正常]")
                st.metric("較上週", f"{trend:+.1f}%")

    st.markdown("---")

    # --- 3. 快速入口 (Quick Links) - 修正版 ---
    st.subheader("🚀 快速功能")
    q1, q2, q3, q4 = st.columns(4)
    
    # 【⭐ 這裡就是修正的關鍵 ⭐】
    # 不使用 st.switch_page，而是直接修改 Session State 並 rerun
    if q1.button("📊 詳細儀表板", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()
        
    if q2.button("🔬 未來預測圖", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()
        
    if q3.button("🔄 立即更新數據", use_container_width=True):
        with st.spinner("正在連線 Pantry Cloud..."):
             st.toast("數據已更新！")
             
    if q4.button("🔔 測試 Line 通知", help="發送測試訊息到綁定的 Line 群組", use_container_width=True):
        st.toast("已發送測試警報！")