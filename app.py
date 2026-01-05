# app.py
import streamlit as st
import time
import pandas as pd
from streamlit_lottie import st_lottie
import concurrent.futures # 【關鍵新增】用於背景執行的函式庫

# 匯入原本的 UI 模組
from app_utils import load_lottiefile
from page_home import show_home_page
from page_dashboard import show_dashboard_page
from page_analysis import show_analysis_page
from page_tutorial import show_tutorial_page

# 匯入後端服務
from model_service import load_resources_and_predict

# --- 0. 頁面設定 ---
st.set_page_config(layout="wide", page_title="智慧電能管家")

# --- 1. 初始化 Session State ---
if "app_ready" not in st.session_state:
    st.session_state.app_ready = False
if "tutorial_complete" not in st.session_state:
    # 如果是第一次來，預設要看導覽
    st.session_state.tutorial_complete = False
if "page" not in st.session_state:
    st.session_state.page = "home"

# 用於儲存 AI 計算結果
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "current_data" not in st.session_state:
    st.session_state.current_data = None

# 【核心修改 1】初始化背景執行緒
# 我們把「未來的結果」存成一個 future 物件，而不直接等待它完成
if "load_future" not in st.session_state:
    # 建立一個執行緒池 (Thread Pool)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # 把重工作提交給它，它會立刻回傳一個 future (代表未來的結果)，不會卡住主程式
    st.session_state.load_future = executor.submit(load_resources_and_predict)
    st.session_state.executor = executor # 保留參照以免被回收

# --- 輔助函式：切換頁面 ---
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# --- 輔助函式：確保資料已載入 ---
def ensure_data_loaded():
    """
    這是一個「檢查站」。
    當使用者要進入主功能時，我們呼叫此函式。
    如果背景還沒跑完，這裡會跳出轉圈圈等待。
    如果背景早就跑完了，這裡會瞬間通過。
    """
    if st.session_state.app_ready:
        return True # 資料已經在手上了

    if "load_future" in st.session_state:
        future = st.session_state.load_future
        
        # 顯示載入畫面 (只有在背景還沒跑完時，使用者才會看到這個)
        if not future.done():
            lottie_json = load_lottiefile("lottiefiles/loading_animation.json")
            placeholder = st.empty()
            with placeholder.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if lottie_json:
                        st_lottie(lottie_json, speed=1, width=300, height=300, key="loading_wait")
                    else:
                        st.spinner("載入中...")
                    st.info("⚡ AI 模型正在做最後衝刺...請稍候")
            
            # 這裡會正式「阻塞 (Block)」，直到背景工作完成
            try:
                pred_df, curr_df = future.result()
            except Exception as e:
                st.error(f"載入失敗: {e}")
                st.stop()
            
            placeholder.empty() # 清除載入動畫
        else:
            # 如果早就做完了，直接拿結果
            pred_df, curr_df = future.result()
            
        # 存入 Session State
        if pred_df is not None:
            st.session_state.prediction_result = pred_df
            st.session_state.current_data = curr_df
            st.session_state.app_ready = True
            return True
        else:
            st.error("啟動失敗：模型服務回傳 None。")
            st.stop()
    return False

# ==========================================
# 🚀 程式主流程 (修改後的邏輯)
# ==========================================

# 1. 如果還沒看完導覽 -> 直接顯示導覽 (不等待資料！)
if not st.session_state.tutorial_complete:
    # 在導覽頁面，Python 會繼續往下跑，而背景執行緒也在同時跑
    show_tutorial_page()
    
    # 注意：如果使用者在導覽頁按了「開始體驗」，tutorial_complete 會變成 True
    # 然後 st.rerun() 會觸發，進入下面的 else區塊

# 2. 如果導覽看完了 (或略過) -> 進入主程式
else:
    # 在進入主程式前，必須過「檢查站」
    # 這時候如果使用者導覽看了很久，資料早就好了，這裡會是 0 秒通過
    if ensure_data_loaded():
        
        # --- 側邊欄導航 ---
        with st.sidebar:
            lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if not lottie_logo: lottie_logo = load_lottiefile("lottiefiles/Intelligent_tour_guide_robot_green.json")     
            if lottie_logo:
                st_lottie(lottie_logo, speed=1, loop=True, quality="high", height=150, key="logo_animation")
            
            st.header("功能選單")
            st.divider()

            current_page = st.session_state.page

            if st.button("🏠 主頁", use_container_width=True, type="primary" if current_page == "home" else "secondary"):
                go_to_page("home")
            
            if st.button("📈 用電儀表板", use_container_width=True, type="primary" if current_page == "dashboard" else "secondary"):
                go_to_page("dashboard")

            if st.button("🔬 AI 決策分析室", use_container_width=True, type="primary" if current_page == "analysis" else "secondary"):
                go_to_page("analysis")
                
            st.divider()
            if st.button("🔄 重新抓取數據"):
                # 重置狀態，讓它重新跑一次 loading
                st.session_state.app_ready = False
                if "load_future" in st.session_state:
                    del st.session_state.load_future
                st.rerun()

        # 頁面路由
        if current_page == "dashboard":
            show_dashboard_page()
        elif current_page == "analysis":
            show_analysis_page()
        else:
            show_home_page()