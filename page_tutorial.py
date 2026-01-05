import streamlit as st
from streamlit_lottie import st_lottie
from app_utils import load_lottiefile

def show_tutorial_page():
    """
    【故事模式】首次使用導覽
    不介紹介面操作，而是介紹核心價值：預警、省錢、滾動修正。
    """
    
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 1

    # 使用欄位將所有內容置中
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        step = st.session_state.tutorial_step
        
        # ==========================================
        # 步驟 1: 價值主張 (Welcome)
        # ==========================================
        if step == 1:
            st.title("歡迎啟用 💡 家庭智慧電管家")
            
            lottie_logo = load_lottiefile("lottiefiles/intelligent_tour_guide_robot.json")
            if lottie_logo:
                st_lottie(lottie_logo, speed=1, loop=True, quality="high", height=250, key="tutorial_logo")
            
            st.markdown("""
            ### 這不是普通的電表查詢 App...
            這是一個會 **24小時主動守護您荷包** 的 AI 能源顧問。
            
            它具備三大核心能力：
            1. **主動預警** (Line 推播)
            2. **預算導航** (講錢不講度數)
            3. **自我進化** (滾動式修正)
            """)
            st.divider()
            
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("略過介紹"):
                    st.session_state.tutorial_complete = True
                    st.rerun()
            with btn_col2:
                if st.button("開始體驗 👉", type="primary"):
                    st.session_state.tutorial_step = 2
                    st.rerun()

        # ==========================================
        # 步驟 2: 必要性 - Line 主動預警
        # ==========================================
        elif step == 2:
            st.title("📱 1. 主動預警，無事不擾")
            
            # [建議] 這裡之後可以放一張 Line 跳出通知的截圖
            try:
                st.image("tutorial_image/tutorial_line_bot.png", caption="當預測即將超支時，AI 會直接傳 Line 給您。")
            except:
                # 如果沒有圖片，用文字模擬
                st.info("💬 Line 通知模擬：\n\n⚠️ **電費警報**\n根據今日用量，預測本月將跨越 $5.0 費率級距！\n建議：今晚冷氣調高 1 度。")

            st.markdown("""
            ### 您不需要天天開 App
            我們知道您很忙。所以，只有當 AI 發現 **「預算即將失控」** 或 **「費率即將跳階」** 時，
            系統才會透過 **Line Bot** 主動通知您。
            
            **👉 讓省電變成一種「被動」的習慣。**
            """)
            st.divider()

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 1
                    st.rerun()
            with btn_col2:
                if st.button("太棒了，還有呢？", type="primary"):
                    st.session_state.tutorial_step = 3
                    st.rerun()

        # ==========================================
        # 步驟 3: 實用性 - 預算導航
        # ==========================================
        elif step == 3:
            st.title("💰 2. 預算導航，拒絕透支")
            
            try:
                st.image("tutorial_image/tutorial_dashboard_budget.png", caption="直觀的預算進度條")
            except:
                st.warning("📊 (這裡將顯示紅/綠色的預算進度條)")

            st.markdown("""
            ### 我們講「錢」，不講「度數」
            看不懂 kWh 沒關係。我們的儀表板直接告訴您：
            
            * **綠色**：目前預測在預算內，請安心使用。
            * **紅色**：警告！依目前趨勢，月底將超支 $500 元。
            
            **👉 就像開車導航一樣，在迷路前就先告訴您該轉彎了。**
            """)
            st.divider()

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 2
                    st.rerun()
            with btn_col2:
                if st.button("最後一個亮點", type="primary"):
                    st.session_state.tutorial_step = 4
                    st.rerun()
        
        # ==========================================
        # 步驟 4: 獨特性 - 滾動式修正
        # ==========================================
        elif step == 4:
            st.title("📈 3. 滾動修正，越用越準")
            
            try:
                st.image("tutorial_image/tutorial_analysis_rolling.png", caption="實線接虛線，每日自動修正")
            except:
                st.info("📈 (這裡將顯示「實線」接「虛線」的預測圖表)")

            st.markdown("""
            ### 這是一個活的 AI 系統
            一般的預測猜完就結束了，但我們的系統每天都在進化。
            
            * **每日校正**：每天凌晨，AI 會吸取昨天的真實數據。
            * **消除誤差**：用「已知」修正「未知」，誤差歸零。
            
            **👉 越接近繳費日，預測準確度無限趨近 100%。**
            """)
            st.divider()

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("上一步"):
                    st.session_state.tutorial_step = 3
                    st.rerun()
            with btn_col2:
                if st.button("✨ 啟動 AI 管家", type="primary"):
                    st.session_state.tutorial_complete = True
                    st.rerun()