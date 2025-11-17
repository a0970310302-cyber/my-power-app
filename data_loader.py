import pandas as pd
import json
import glob  # 用於尋找所有匹配的檔案
import os    # 用於處理檔案和資料夾路徑
import re    # 匯入正規表示式模組

def load_single_file(json_path):
    """
    讀取「單一」JSON 檔案並進行基礎清理。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        data_list = raw_data.get("listAMIBase15MinData", [])
        if not data_list:
            return None # 檔案是空的或格式不符

        df = pd.DataFrame(data_list)
        
        # 我們只需要 "time" 和 "power" 欄位
        df = df[["time", "power"]]

        # --- 從檔名解析「日期」，僅支援 YYYY-MM-DD 格式 ---
        file_name = os.path.basename(json_path)
        try:
            date_str = None
            # 1. 嘗試匹配 YYYY-MM-DD 格式
            match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
            if match:
                date_str = match.group(1)
                file_date = pd.to_datetime(date_str, format='%Y-%m-%d').strftime('%Y-%m-%d')
            else:
                # 2. 如果匹配失敗，觸發 Exception
                raise ValueError("檔名中未包含 YYYY-MM-DD 格式的日期")
            
        except Exception as e:
            # 如果檔名不符規定，使用檔案的最後修改時間 (備案)
            stat = os.stat(json_path)
            file_date = pd.to_datetime(stat.st_mtime, unit='s').strftime('%Y-%m-%d')
            print(f"警告：無法從檔名 {file_name} 解析日期 ({e})。改用檔案修改日期 {file_date}")
        # --- 修改結束 ---

        # 合併日期和時間，建立完整的時間戳
        df['timestamp'] = pd.to_datetime(file_date + " " + df['time'])
        df = df.rename(columns={"power": "power_kW"})
        
        # 將 power 轉換為數值，並將 timestamp 設為索引
        df['power_kW'] = pd.to_numeric(df['power_kW'], errors='coerce')
        df = df.set_index('timestamp')
        
        return df[['power_kW']]
        
    except Exception as e:
        print(f"處理檔案 {json_path} 失敗: {e}")
        return None

def load_all_history_data():
    """
    【關鍵函式】
    讀取「目前資料夾」中所有的 "taipower_raw_*.json" 檔案，
    並合併成一個完整的歷史數據 DataFrame。
    """
    
    # 1. 找出所有 "taipower_raw_*.json" 檔案的路徑
    #    ( "taipower_raw_*.json" 會找到所有開頭是 taipower_raw_ 結尾是 .json 的檔案)
    all_json_files = glob.glob("taipower_raw_*.json")
    
    if not all_json_files:
        # 如果找不到任何檔案
        print(f"錯誤：在目前資料夾中找不到任何 'taipower_raw_*.json' 檔案。")
        print("請確認爬蟲下載的 JSON 檔案與 app.py 放在同一個資料夾。")
        return pd.DataFrame(columns=['power_kW']) # 回傳一個空的 DataFrame
    
    # 2. 逐一讀取並清理
    all_dataframes = []
    print(f"正在載入 {len(all_json_files)} 個 JSON 檔案...")
    for file_path in all_json_files:
        df = load_single_file(file_path)
        if df is not None:
            all_dataframes.append(df)
            
    if not all_dataframes:
        print("警告：所有 JSON 檔案都無法成功解析。")
        return pd.DataFrame(columns=['power_kW']) # 回傳一個空的 DataFrame

    # 3. 合併所有 DataFrame
    full_history_df = pd.concat(all_dataframes)
    
    # 4. 排序並移除重複的索引 (以防萬一)
    full_history_df = full_history_df.sort_index()
    full_history_df = full_history_df[~full_history_df.index.duplicated(keep='first')]
    
    print("--- 完整歷史數據載入成功 ---")
    return full_history_df

# 這段程式碼允許您單獨執行 `python data_loader.py` 來測試是否成功
if __name__ == "__main__":
    try:
        full_data = load_all_history_data()
        
        if full_data.empty:
            print("\n--- 資料載入測試完成，但未載入任何有效數據 ---")
        else:
            print("\n--- 資料載入測試成功 ---")
            print(full_data.head()) # 顯示最早的 5 筆
            print(full_data.tail()) # 顯示最新的 5 筆
            print(f"\n總共 {len(full_data)} 筆資料，從 {full_data.index.min()} 到 {full_data.index.max()}")
            print(f"\n資料描述：\n{full_data.describe()}")
        
    except Exception as e:
        print(f"\n--- 資料載入測試失敗 ---")
        print(e)

