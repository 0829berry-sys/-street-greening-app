# -*- coding: utf-8 -*-
"""
==================================================================================
街道非正式綠化（盆栽佔用）現場調查分析系統  v3
Streamlit + OpenCV 電腦視覺應用程式
==================================================================================

【安裝相依套件】
    pip install streamlit opencv-python numpy pandas pillow matplotlib openpyxl xlsxwriter

【啟動方式】
    streamlit run app.py
    （若出現 'streamlit' 無法辨識，改用：python -m streamlit run app.py）

【v3 更新重點】
    1. 側邊欄：照片上傳移到最上方，點位編號輸入後立即顯示 ArUco 比對結果；
       移除「盆栽主要類型」「盆栽擺放位置（相對位置）」；新增名詞說明區塊。
    2. 影像分析參數每一項都附上 (i) 提示圖示（滑鼠移過去顯示說明）。
    3. 主畫面新增「筆刷編輯」工具：可手動增補或移除被誤判的綠化區域，
       重新計算後才能「完成編輯」並儲存，避免存到編輯前的舊資料。
    4. 新增「歷史紀錄查詢」，可依捷運站或紀錄日期排序/分組瀏覽過去儲存的點位。
    5. 擺放形式判定分類簡化，避免相近的立面高度組合被分到不同類別。
    6. 資料庫改為本機自動存檔（CSV，相對路徑），程式重新開啟時自動讀回舊資料；
       同時將上傳照片另存於本機資料夾，供之後於歷史紀錄中回看。
==================================================================================
"""

import io
import os
import zipfile
import uuid
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# st.dialog（彈出視窗）是較新版本 Streamlit 才有的功能，這裡做版本相容處理
DIALOG_DECORATOR = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
DIALOG_SUPPORTED = DIALOG_DECORATOR is not None

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    IMAGE_COORDS_AVAILABLE = True
except ImportError:
    IMAGE_COORDS_AVAILABLE = False

matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------------
# 全域設定
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="街道非正式綠化（盆栽佔用）調查分析系統",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(APP_DIR, "survey_database.csv")
PHOTOS_DIR = os.path.join(APP_DIR, "survey_photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)

DATAFRAME_COLUMNS = [
    "捷運站",
    "點位編號",
    "ArUco編號",
    "ArUco偵測ID",
    "編號比對結果",
    "拍照距離(m)",
    "擺放位置",
    "盆栽擺放型態",
    "落地擺放高度(cm)",
    "吊掛擺放高度(cm)",
    "向上擺放高度(cm)",
    "盆栽占用深度(cm)",
    "落地擺放數量",
    "吊掛擺放數量",
    "向上擺放數量",
    "空盆栽數量",
    "擺放數量合計(不含空盆)",
    "附註",
    "ArUco邊長(cm)",
    "比例尺來源",
    "像素/公分比例尺",
    "AI辨識綠化面積(m2)",
    "左區高度(cm)",
    "中區高度(cm)",
    "右區高度(cm)",
    "高低型態判定",
    "照片檔名",
    "紀錄時間",
]

NUMERIC_COLS_FOR_STATS = [
    "拍照距離(m)",
    "落地擺放高度(cm)",
    "吊掛擺放高度(cm)",
    "向上擺放高度(cm)",
    "盆栽占用深度(cm)",
    "落地擺放數量",
    "吊掛擺放數量",
    "向上擺放數量",
    "空盆栽數量",
    "擺放數量合計(不含空盆)",
    "AI辨識綠化面積(m2)",
    "左區高度(cm)",
    "中區高度(cm)",
    "右區高度(cm)",
]

CATEGORICAL_COLS_FOR_STATS = ["擺放位置", "盆栽擺放型態", "高低型態判定"]

# 常見 ArUco 字典，將依序嘗試偵測
ARUCO_DICT_CANDIDATES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}

HEIGHT_LEVEL_THRESHOLD_DEFAULT = 5.0  # 公分，判斷「高/低」是否有顯著落差的閾值

LOCATION_OPTIONS = ["路邊", "門口", "路口轉角", "騎樓", "其他"]
ARRANGEMENT_OPTIONS = ["橫向擺放", "直列擺放", "群聚擺放", "單盆擺放"]

# 批次匯入 Excel 範本欄位：只包含「人工填寫」的屬性，不含 AI 計算出來的欄位（那些要等上傳照片、執行分析後才會產生）
# 「ArUco編號」是貼在該點位盆栽上的 ArUco 標記號碼，跟「點位編號」（你自己的命名方式）分開存放，
# 是為了讓「批次上傳照片、依 ArUco 自動比對」功能可以運作，即使你的點位編號不是單純數字也沒關係。
TEMPLATE_COLUMNS = [
    "捷運站", "點位編號", "ArUco編號", "拍照距離(m)", "擺放位置", "盆栽擺放型態",
    "落地擺放高度(cm)", "吊掛擺放高度(cm)", "向上擺放高度(cm)",
    "盆栽占用深度(cm)",
    "落地擺放數量", "吊掛擺放數量", "向上擺放數量", "空盆栽數量",
    "ArUco邊長(cm)", "附註",
]

CANVAS_MAX_WIDTH = 850  # 遮罩編輯預覽圖的最大寬度（像素）


# ----------------------------------------------------------------------------
# 本機資料存取（CSV，使用相對於程式檔案的路徑，Windows / Mac 皆可直接使用）
# ----------------------------------------------------------------------------
def load_local_data():
    if os.path.exists(DATA_FILE_PATH):
        try:
            df = pd.read_csv(DATA_FILE_PATH, dtype=str, keep_default_na=False)
            # 補齊欄位（若舊檔案缺少新欄位）並還原數值型別
            for col in DATAFRAME_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[DATAFRAME_COLUMNS]
            for col in NUMERIC_COLS_FOR_STATS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=DATAFRAME_COLUMNS)
    return pd.DataFrame(columns=DATAFRAME_COLUMNS)


def save_local_data(df):
    try:
        df.to_csv(DATA_FILE_PATH, index=False, encoding="utf-8-sig")
        return True
    except Exception:
        return False


def save_photo_file(image_bgr, point_id, timestamp_str):
    safe_id = "".join(c for c in str(point_id) if c.isalnum() or c in ("-", "_")) or "point"
    # 用微秒級時間戳＋隨機字串，確保檔名絕對不會撞名（避免批次處理時，
    # 短時間內存很多張照片、或不同捷運站剛好用了相同點位編號，導致互相覆蓋）
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{safe_id}_{unique_token}.jpg"
    filepath = os.path.join(PHOTOS_DIR, filename)
    if os.path.exists(filepath):
        filename = f"{safe_id}_{unique_token}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(PHOTOS_DIR, filename)
    try:
        ok = imwrite_unicode(filepath, image_bgr)
        return filename if ok else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# 完整備份／還原（ZIP，內含資料表 CSV ＋ 所有照片）
# ----------------------------------------------------------------------------
def build_full_backup_zip(df):
    """把目前資料表與所有已存照片打包成一個 ZIP 檔案（bytes）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("survey_database.csv", df.to_csv(index=False).encode("utf-8-sig"))
        if os.path.isdir(PHOTOS_DIR):
            for filename in os.listdir(PHOTOS_DIR):
                filepath = os.path.join(PHOTOS_DIR, filename)
                if os.path.isfile(filepath):
                    zf.write(filepath, arcname=f"survey_photos/{filename}")
    buffer.seek(0)
    return buffer.getvalue()


def restore_from_backup_zip(zip_bytes, mode="replace"):
    """
    從備份 ZIP 還原資料表與照片。
    mode="replace"：完全取代目前資料與照片
    mode="merge"：把備份中的資料附加到目前資料表後面（照片一律複製進來，同檔名會被覆蓋）
    回傳 (success: bool, message: str)
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            namelist = zf.namelist()
            if "survey_database.csv" not in namelist:
                return False, "這個 ZIP 檔案裡找不到 survey_database.csv，不是本程式匯出的備份檔。"

            csv_bytes = zf.read("survey_database.csv")
            restored_df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
            for col in DATAFRAME_COLUMNS:
                if col not in restored_df.columns:
                    restored_df[col] = ""
            restored_df = restored_df[DATAFRAME_COLUMNS]
            for col in NUMERIC_COLS_FOR_STATS:
                restored_df[col] = pd.to_numeric(restored_df[col], errors="coerce")

            # 還原照片
            os.makedirs(PHOTOS_DIR, exist_ok=True)
            photo_count = 0
            for name in namelist:
                if name.startswith("survey_photos/") and not name.endswith("/"):
                    target_filename = os.path.basename(name)
                    if not target_filename:
                        continue
                    target_path = os.path.join(PHOTOS_DIR, target_filename)
                    with open(target_path, "wb") as f_out:
                        f_out.write(zf.read(name))
                    photo_count += 1

            if mode == "replace":
                final_df = restored_df
            else:
                final_df = pd.concat(
                    [st.session_state.dataframe, restored_df], ignore_index=True
                )

            st.session_state.dataframe = final_df
            save_local_data(final_df)
            return True, f"還原完成：共 {len(restored_df)} 筆資料、{photo_count} 張照片。"
    except zipfile.BadZipFile:
        return False, "這個檔案不是有效的 ZIP 格式，請確認上傳的是本程式匯出的備份檔。"
    except Exception as e:
        return False, f"還原時發生錯誤：{e}"


# ----------------------------------------------------------------------------
# Session State 初始化
# ----------------------------------------------------------------------------
def init_session_state():
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = load_local_data()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "current_mask" not in st.session_state:
        st.session_state.current_mask = None
    if "current_image_bgr" not in st.session_state:
        st.session_state.current_image_bgr = None
    if "current_pixels_per_cm" not in st.session_state:
        st.session_state.current_pixels_per_cm = None
    if "editing_mode" not in st.session_state:
        st.session_state.editing_mode = False
    if "canvas_key_counter" not in st.session_state:
        st.session_state.canvas_key_counter = 0
    if "mask_before_editing" not in st.session_state:
        st.session_state.mask_before_editing = None
    if "rect_first_corner" not in st.session_state:
        st.session_state.rect_first_corner = None
    if "rect_second_corner" not in st.session_state:
        st.session_state.rect_second_corner = None
    if "last_click_coords" not in st.session_state:
        st.session_state.last_click_coords = None
    if "height_threshold_cm" not in st.session_state:
        st.session_state.height_threshold_cm = HEIGHT_LEVEL_THRESHOLD_DEFAULT
    if "pending_records" not in st.session_state:
        st.session_state.pending_records = []
    if "pending_id_counter" not in st.session_state:
        st.session_state.pending_id_counter = 0
    if "active_batch_pending_id" not in st.session_state:
        st.session_state.active_batch_pending_id = None
    if "active_history_edit_index" not in st.session_state:
        st.session_state.active_history_edit_index = None
    if "show_manual_form" not in st.session_state:
        st.session_state.show_manual_form = True


# ----------------------------------------------------------------------------
# ArUco 偵測（僅取 ID 清單，供點位編號比對用，不需比例尺）
# ----------------------------------------------------------------------------
def detect_aruco_ids_only(image_bgr):
    """快速偵測畫面中所有 ArUco Marker 的 ID，不做比例尺換算。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()

    for dict_name, dict_id in ARUCO_DICT_CANDIDATES.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        _corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            return [int(x) for x in ids.flatten()], dict_name

    return [], None


def build_aruco_match_message(point_id, ids_found):
    """依偵測到的 ArUco ID 清單與手動輸入之點位編號，組出比對結果文字。"""
    if not ids_found:
        return "未偵測到 ArUco 標記", "未偵測到"

    id_str = "、".join(str(i) for i in ids_found)
    # point_id 可能來自表單文字輸入（str），也可能來自 Excel 匯入（int/float/numpy 數值型別），
    # 這裡一律先轉成字串再處理，避免數值型別沒有 .strip() 方法而出錯。
    point_id_clean = "" if point_id is None else str(point_id).strip()
    if point_id_clean.endswith(".0"):
        point_id_clean = point_id_clean[:-2]  # Excel 數字欄位可能變成 "1.0" 這種形式，去除多餘的 .0

    try:
        point_id_num = int(point_id_clean)
    except (ValueError, TypeError):
        point_id_num = None

    if point_id_num is not None and point_id_num in ids_found:
        return f"已偵測到 ArUco（ID：{id_str}） ✅ 與手動編號相符", "相符"
    elif point_id_num is not None:
        return f"已偵測到 ArUco（ID：{id_str}） ⚠️ 與手動編號不符", "不符"
    else:
        return f"已偵測到 ArUco（ID：{id_str}） ℹ️ 手動編號非純數字，無法自動比對", "無法比對"


# ----------------------------------------------------------------------------
# ArUco 偵測與比例尺計算（OpenCV 4.7+ 新版 API：ArucoDetector）
# ----------------------------------------------------------------------------
def detect_aruco_marker(image_bgr, marker_real_length_cm):
    """
    嘗試以多種常見 ArUco 字典偵測畫面中的 Marker，並以第一個偵測到的 marker 做比例尺校正。
    成功則回傳：
        (annotated_image, pixels_per_cm, marker_center_xy, dict_name_used, marker_corners, marker_id)
    失敗則回傳：
        (annotated_image(原圖), None, None, None, None, None)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    annotated = image_bgr.copy()

    detector_params = cv2.aruco.DetectorParameters()

    for dict_name, dict_id in ARUCO_DICT_CANDIDATES.items():
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            marker_corners = corners[0].reshape(4, 2)
            marker_id = int(ids.flatten()[0])

            side_lengths_px = [
                np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
                for i in range(4)
            ]
            avg_side_px = float(np.mean(side_lengths_px))
            pixels_per_cm = avg_side_px / marker_real_length_cm

            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            center_xy = marker_corners.mean(axis=0)
            cv2.circle(annotated, (int(center_xy[0]), int(center_xy[1])), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"{dict_name} ID:{marker_id} | {avg_side_px:.1f}px = {marker_real_length_cm}cm",
                (int(marker_corners[0][0]), max(int(marker_corners[0][1]) - 15, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )

            return annotated, pixels_per_cm, center_xy, dict_name, marker_corners, marker_id

    return annotated, None, None, None, None, None


# ----------------------------------------------------------------------------
# 綠色植栽遮罩擷取
# ----------------------------------------------------------------------------
def extract_green_mask_hsv(image_bgr, h_low, h_high, s_low, v_low):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def extract_green_mask_exg(image_bgr, exg_threshold):
    img = image_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, exg_threshold, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean


def compute_green_area_m2(mask, pixels_per_cm):
    green_pixel_count = int(np.count_nonzero(mask))
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return None, green_pixel_count
    area_cm2 = green_pixel_count / (pixels_per_cm ** 2)
    area_m2 = area_cm2 / 10000.0
    return area_m2, green_pixel_count


# ----------------------------------------------------------------------------
# 立面輪廓（Upper Envelope）左/中/右三區高度分析
# ----------------------------------------------------------------------------
def analyze_zone_profile(mask, pixels_per_cm):
    h, w = mask.shape[:2]
    col_has_green = np.any(mask > 0, axis=0)
    valid_x = np.where(col_has_green)[0]

    if len(valid_x) == 0:
        return {"left": None, "mid": None, "right": None}, [], None

    x_min, x_max = int(valid_x.min()), int(valid_x.max())
    span = x_max - x_min
    x_split1 = x_min + span // 3
    x_split2 = x_min + 2 * span // 3

    envelope_points = []
    for x in valid_x:
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        y_top = int(ys.min())
        envelope_points.append((int(x), y_top))

    def zone_avg_height_cm(x_lo, x_hi):
        ys_in_zone = [y for (x, y) in envelope_points if x_lo <= x <= x_hi]
        if len(ys_in_zone) == 0:
            return None
        avg_y_top = float(np.mean(ys_in_zone))
        height_px = h - avg_y_top
        if pixels_per_cm is None or pixels_per_cm <= 0:
            return None
        return round(height_px / pixels_per_cm, 1)

    zone_heights_cm = {
        "left": zone_avg_height_cm(x_min, x_split1),
        "mid": zone_avg_height_cm(x_split1, x_split2),
        "right": zone_avg_height_cm(x_split2, x_max),
    }

    return zone_heights_cm, envelope_points, (x_min, x_split1, x_split2, x_max)


def classify_zone_shape(zone_heights_cm, threshold_cm=HEIGHT_LEVEL_THRESHOLD_DEFAULT):
    """
    依左/中/右三區高度關係，自動分類擺放立面形式（固定為 5 類，避免相近組合被拆成
    不同類別）：
        - 齊平型：整體高度差小於閾值，或中間高度落在左右連線的合理範圍內
        - 凹型（高低高）：中間明顯低於左右兩側連線
        - 拱型（低高低）：中間明顯高於左右兩側連線
        - 遞增型（左低右高）：整體呈現由左至右升高的單調趨勢
        - 遞減型（左高右低）：整體呈現由左至右降低的單調趨勢
    """
    hl, hm, hr = zone_heights_cm.get("left"), zone_heights_cm.get("mid"), zone_heights_cm.get("right")
    if hl is None or hm is None or hr is None:
        return "資料不足（無法判定）"

    vals = [hl, hm, hr]
    if max(vals) - min(vals) < threshold_cm:
        return "齊平型"

    expected_mid = (hl + hr) / 2.0
    mid_deviation = hm - expected_mid
    diff_lr = hl - hr

    if mid_deviation > threshold_cm:
        return "拱型（低高低）"
    if mid_deviation < -threshold_cm:
        return "凹型（高低高）"
    if diff_lr > threshold_cm:
        return "遞減型（左高右低）"
    if diff_lr < -threshold_cm:
        return "遞增型（左低右高）"
    return "齊平型"


# ----------------------------------------------------------------------------
# 繪製立面輪廓與分區示意圖
# ----------------------------------------------------------------------------
def draw_envelope_visualization(image_bgr, envelope_points, x_bounds, zone_heights_cm):
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    if x_bounds is None or len(envelope_points) == 0:
        cv2.putText(vis, "未偵測到綠色植栽區域", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    x_min, x_split1, x_split2, x_max = x_bounds

    pts = np.array(sorted(envelope_points, key=lambda p: p[0]), dtype=np.int32)
    for i in range(len(pts) - 1):
        cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 140, 255), 2)

    cv2.line(vis, (x_split1, 0), (x_split1, h), (255, 255, 0), 2)
    cv2.line(vis, (x_split2, 0), (x_split2, h), (255, 255, 0), 2)

    labels = [
        ("左區", zone_heights_cm.get("left"), (x_min + x_split1) // 2),
        ("中區", zone_heights_cm.get("mid"), (x_split1 + x_split2) // 2),
        ("右區", zone_heights_cm.get("right"), (x_split2 + x_max) // 2),
    ]
    for name, val, cx in labels:
        text = f"{name}: {val}cm" if val is not None else f"{name}: N/A"
        cv2.putText(vis, text, (max(cx - 60, 5), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return vis


# ----------------------------------------------------------------------------
# 影像格式轉換工具
# ----------------------------------------------------------------------------
def pil_to_bgr(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb_for_display(bgr_image):
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def imwrite_unicode(filepath, image_bgr, ext=".jpg"):
    """
    Windows 上如果檔案路徑含有中文字元（例如帳號、資料夾名稱是中文），
    cv2.imwrite / cv2.imread 常會「靜默失敗」──不會丟出例外，只是直接回傳
    False / None，導致照片實際上沒寫入或讀不回來。這裡改用「先編碼成 bytes、
    再用 Python 內建檔案寫入」的方式繞過這個已知限制。
    """
    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        return False
    with open(filepath, "wb") as f:
        f.write(encoded.tobytes())
    return True


def imread_unicode(filepath):
    """對應 imwrite_unicode 的讀取版本，同樣避開中文路徑的問題。"""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_for_canvas(image_bgr, max_width=CANVAS_MAX_WIDTH):
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr.copy()
    scale = max_width / w
    resized = cv2.resize(image_bgr, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized


# ----------------------------------------------------------------------------
# 匯出工具
# ----------------------------------------------------------------------------
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_excel_bytes(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="調查資料")
    buffer.seek(0)
    return buffer.getvalue()


def build_template_excel_bytes():
    """產生批次匯入用的 Excel 範本，含一行範例與一張欄位選項說明表。"""
    example_row = {
        "捷運站": "忠孝復興站", "點位編號": "A01", "ArUco編號": 23, "拍照距離(m)": 3.0,
        "擺放位置": "路邊", "盆栽擺放型態": "橫向擺放",
        "落地擺放高度(cm)": 60, "吊掛擺放高度(cm)": 0, "向上擺放高度(cm)": 0,
        "盆栽占用深度(cm)": 40,
        "落地擺放數量": 5, "吊掛擺放數量": 0, "向上擺放數量": 0, "空盆栽數量": 1,
        "ArUco邊長(cm)": 20, "附註": "範例列，請刪除後填入自己的資料",
    }
    template_df = pd.DataFrame([example_row], columns=TEMPLATE_COLUMNS)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="點位資料")
        options_df = pd.DataFrame({
            "擺放位置可填值": pd.Series(LOCATION_OPTIONS),
            "盆栽擺放型態可填值": pd.Series(ARRANGEMENT_OPTIONS),
        })
        options_df.to_excel(writer, index=False, sheet_name="欄位選項說明")
    buffer.seek(0)
    return buffer.getvalue()


def parse_template_excel(file_bytes):
    """
    解析使用者上傳的 Excel 範本，回傳 (records: list[dict], error: str or None)。
    每筆 record 只含 TEMPLATE_COLUMNS 欄位（缺的欄位補空/補 0）。
    """
    try:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="點位資料")
        except Exception:
            df = pd.read_excel(io.BytesIO(file_bytes))  # 找不到指定分頁名稱時，退回讀第一個分頁

        if "點位編號" not in df.columns:
            return [], "找不到「點位編號」欄位，請確認是用本程式提供的範本檔案。"

        for col in TEMPLATE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        numeric_template_cols = [
            "拍照距離(m)", "ArUco編號", "落地擺放高度(cm)", "吊掛擺放高度(cm)", "向上擺放高度(cm)",
            "盆栽占用深度(cm)", "落地擺放數量", "吊掛擺放數量", "向上擺放數量",
            "空盆栽數量", "ArUco邊長(cm)",
        ]
        for col in numeric_template_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = df.dropna(subset=["點位編號"])
        df["點位編號"] = df["點位編號"].apply(
            lambda v: str(v)[:-2] if isinstance(v, float) and str(v).endswith(".0") else str(v)
        ).str.strip()
        df = df[df["點位編號"] != ""]

        records = df[TEMPLATE_COLUMNS].to_dict(orient="records")
        return records, None
    except Exception as e:
        return [], f"讀取 Excel 檔案失敗：{e}"


# ----------------------------------------------------------------------------
# 從目前的 mask 與比例尺，重新計算完整分析結果（初次分析與筆刷編輯後皆呼叫此函式）
# ----------------------------------------------------------------------------
def recompute_full_result(mask, pixels_per_cm, height_threshold_cm):
    area_m2, green_pixel_count = compute_green_area_m2(mask, pixels_per_cm)
    zone_heights_cm, envelope_points, x_bounds = analyze_zone_profile(mask, pixels_per_cm)
    shape_label = classify_zone_shape(zone_heights_cm, threshold_cm=height_threshold_cm)
    return {
        "area_m2": area_m2,
        "green_pixel_count": green_pixel_count,
        "zone_heights_cm": zone_heights_cm,
        "envelope_points": envelope_points,
        "x_bounds": x_bounds,
        "shape_label": shape_label,
    }


def render_result_visuals(image_bgr, mask, computed):
    mask_overlay = image_bgr.copy()
    mask_overlay[mask > 0] = (0, 255, 0)
    mask_blend = cv2.addWeighted(image_bgr, 0.5, mask_overlay, 0.5, 0)
    envelope_vis = draw_envelope_visualization(
        image_bgr, computed["envelope_points"], computed["x_bounds"], computed["zone_heights_cm"]
    )

    st.subheader("🖼️ 影像分析視覺化比對")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(bgr_to_rgb_for_display(image_bgr), caption="原圖", use_container_width=True)
    with col2:
        st.image(bgr_to_rgb_for_display(mask_blend), caption="綠色植栽遮罩疊圖", use_container_width=True)
    with col3:
        st.image(bgr_to_rgb_for_display(envelope_vis), caption="立面上緣輪廓 + 左中右分區", use_container_width=True)

    st.subheader("📊 分析結果摘要")
    zh = computed["zone_heights_cm"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI 辨識綠化面積 (m²)", f"{computed['area_m2']:.4f}" if computed["area_m2"] is not None else "N/A")
    m2.metric("擺放形式判定", computed["shape_label"])
    m3.metric("綠色像素數", f"{computed['green_pixel_count']:,}")
    m4.metric("左/中/右高度(cm)", f"{zh.get('left')}/{zh.get('mid')}/{zh.get('right')}")


# ----------------------------------------------------------------------------
# 筆刷編輯視窗（彈出對話框；直接對「遮罩」本身進行編輯）
# ----------------------------------------------------------------------------
def _render_finish_revert_cancel_buttons():
    """三個編輯階段共用的收尾按鈕：復原本次編輯／完成編輯／放棄並關閉。"""
    revert_col, finish_col, cancel_col = st.columns(3)
    with revert_col:
        if st.button("↩️ 復原本次所有編輯", use_container_width=True):
            st.session_state.current_mask = st.session_state.mask_before_editing.copy()
            st.session_state.rect_first_corner = None
            st.session_state.rect_second_corner = None
            st.rerun()
    with finish_col:
        if st.button("✅ 完成編輯", type="primary", use_container_width=True):
            final_computed = recompute_full_result(
                st.session_state.current_mask,
                st.session_state.current_pixels_per_cm,
                st.session_state.height_threshold_cm,
            )
            lr = st.session_state.last_result
            lr["AI辨識綠化面積(m2)"] = round(final_computed["area_m2"], 4) if final_computed["area_m2"] is not None else None
            lr["左區高度(cm)"] = final_computed["zone_heights_cm"].get("left")
            lr["中區高度(cm)"] = final_computed["zone_heights_cm"].get("mid")
            lr["右區高度(cm)"] = final_computed["zone_heights_cm"].get("right")
            lr["高低型態判定"] = final_computed["shape_label"]
            st.session_state.last_result = lr
            st.session_state.editing_mode = False
            st.session_state.rect_first_corner = None
            st.session_state.rect_second_corner = None
            st.rerun()
    with cancel_col:
        if st.button("❌ 放棄本次編輯並關閉", use_container_width=True):
            st.session_state.current_mask = st.session_state.mask_before_editing.copy()
            st.session_state.editing_mode = False
            st.session_state.rect_first_corner = None
            st.session_state.rect_second_corner = None
            st.rerun()


def _render_mask_editor_click():
    """滑鼠點兩下框選矩形（主要方式，需要 streamlit-image-coordinates 套件）。"""
    st.info(
        "🖱️ 在下方照片上**點一下**設定矩形的一個角，**再點一下**設定對角，就會框出矩形。"
        "選擇要「新增」還是「移除」，按「➕ 套用這個矩形」即可疊加到遮罩上；"
        "可以連續點選、套用多個矩形來組合出想要的形狀。"
    )

    mode = st.radio("這個矩形要做什麼", ["新增到綠化面積（綠框）", "從綠化面積移除（紅框）"], horizontal=True, key="rect_mode")

    overlay = st.session_state.current_image_bgr.copy()
    overlay[st.session_state.current_mask > 0] = (0, 255, 0)
    blend = cv2.addWeighted(st.session_state.current_image_bgr, 0.5, overlay, 0.5, 0)
    preview_base = resize_for_canvas(blend, max_width=CANVAS_MAX_WIDTH)
    disp_h, disp_w = preview_base.shape[:2]
    orig_h, orig_w = st.session_state.current_mask.shape[:2]
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    rect_color = (0, 255, 0) if mode.startswith("新增") else (0, 0, 255)
    first_corner = st.session_state.get("rect_first_corner")
    second_corner = st.session_state.get("rect_second_corner")

    preview = preview_base.copy()
    if first_corner is not None:
        cv2.drawMarker(preview, first_corner, (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3)
    if first_corner is not None and second_corner is not None:
        rx0, rx1 = sorted([first_corner[0], second_corner[0]])
        ry0, ry1 = sorted([first_corner[1], second_corner[1]])
        cv2.rectangle(preview, (rx0, ry0), (rx1, ry1), rect_color, 3)

    pil_preview = Image.fromarray(bgr_to_rgb_for_display(preview))
    coords = streamlit_image_coordinates(pil_preview, key=f"mask_click_{st.session_state.canvas_key_counter}")

    if coords is not None:
        current_click = (int(coords["x"]), int(coords["y"]))
        if current_click != st.session_state.get("last_click_coords"):
            st.session_state.last_click_coords = current_click
            if first_corner is None or second_corner is not None:
                # 還沒開始選，或上一個矩形已經選滿了 → 開始一個新的矩形
                st.session_state.rect_first_corner = current_click
                st.session_state.rect_second_corner = None
            else:
                st.session_state.rect_second_corner = current_click
            st.rerun()

    if first_corner is not None and second_corner is not None:
        rx0, rx1 = sorted([first_corner[0], second_corner[0]])
        ry0, ry1 = sorted([first_corner[1], second_corner[1]])
        apply_col, reset_col = st.columns(2)
        with apply_col:
            if st.button("➕ 套用這個矩形", type="primary", use_container_width=True):
                x0, x1 = int(rx0 * scale_x), int(rx1 * scale_x)
                y0, y1 = int(ry0 * scale_y), int(ry1 * scale_y)
                new_mask = st.session_state.current_mask.copy()
                if mode.startswith("新增"):
                    new_mask[y0:y1, x0:x1] = 255
                else:
                    new_mask[y0:y1, x0:x1] = 0
                st.session_state.current_mask = new_mask
                st.session_state.rect_first_corner = None
                st.session_state.rect_second_corner = None
                st.session_state.canvas_key_counter += 1
                st.rerun()
        with reset_col:
            if st.button("🔄 重新選取矩形", use_container_width=True):
                st.session_state.rect_first_corner = None
                st.session_state.rect_second_corner = None
                st.rerun()
    elif first_corner is not None:
        st.caption("已設定第一個角，請在照片上點第二下設定對角。")

    st.divider()
    _render_finish_revert_cancel_buttons()


def _render_mask_editor_sliders():
    """滑桿框選矩形（備用方式：滑鼠點選套件不可用時自動改用這個）。"""
    st.info(
        "🖌️ 用下面的滑桿框出一個矩形區域，選擇要「新增」還是「移除」，按「套用」即可疊加到遮罩上；"
        "可以連續套用多個矩形來組合出想要的形狀。"
    )

    overlay = st.session_state.current_image_bgr.copy()
    overlay[st.session_state.current_mask > 0] = (0, 255, 0)
    blend = cv2.addWeighted(st.session_state.current_image_bgr, 0.5, overlay, 0.5, 0)
    preview_base = resize_for_canvas(blend, max_width=CANVAS_MAX_WIDTH)
    disp_h, disp_w = preview_base.shape[:2]
    orig_h, orig_w = st.session_state.current_mask.shape[:2]
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    mode = st.radio("這個矩形要做什麼", ["新增到綠化面積（綠框）", "從綠化面積移除（紅框）"], horizontal=True, key="rect_mode")

    c1, c2 = st.columns(2)
    with c1:
        rect_x = st.slider("左邊界 X", 0, max(disp_w - 1, 1), int(disp_w * 0.3), key="rect_x")
        rect_w = st.slider("寬度", 1, disp_w, max(int(disp_w * 0.2), 1), key="rect_w")
    with c2:
        rect_y = st.slider("上邊界 Y", 0, max(disp_h - 1, 1), int(disp_h * 0.3), key="rect_y")
        rect_h = st.slider("高度", 1, disp_h, max(int(disp_h * 0.2), 1), key="rect_h")

    rect_x2 = min(rect_x + rect_w, disp_w)
    rect_y2 = min(rect_y + rect_h, disp_h)
    rect_color = (0, 255, 0) if mode.startswith("新增") else (0, 0, 255)

    preview = preview_base.copy()
    cv2.rectangle(preview, (rect_x, rect_y), (rect_x2, rect_y2), rect_color, 3)
    st.image(
        bgr_to_rgb_for_display(preview),
        caption="目前遮罩＋這次要套用的矩形範圍（框線顏色代表新增／移除）",
        use_container_width=True,
    )

    if st.button("➕ 套用這個矩形", type="primary", use_container_width=True):
        x0, x1 = int(rect_x * scale_x), int(rect_x2 * scale_x)
        y0, y1 = int(rect_y * scale_y), int(rect_y2 * scale_y)
        new_mask = st.session_state.current_mask.copy()
        if mode.startswith("新增"):
            new_mask[y0:y1, x0:x1] = 255
        else:
            new_mask[y0:y1, x0:x1] = 0
        st.session_state.current_mask = new_mask
        st.rerun()

    st.divider()
    _render_finish_revert_cancel_buttons()


def _render_mask_editor_body():
    if IMAGE_COORDS_AVAILABLE:
        _render_mask_editor_click()
    else:
        st.warning(
            "尚未安裝滑鼠圈選所需套件，暫時使用滑桿版本。"
            "如果想改用滑鼠點選，請於終端機執行：pip install streamlit-image-coordinates 後重新啟動程式。"
        )
        _render_mask_editor_sliders()


if DIALOG_SUPPORTED:
    @DIALOG_DECORATOR("✏️ 編輯遮罩", width="large")
    def mask_editor_dialog():
        _render_mask_editor_body()
else:
    def mask_editor_dialog():
        st.warning(
            "目前 Streamlit 版本不支援彈出視窗，請執行「pip install --upgrade streamlit」升級後即可使用彈出式編輯視窗。"
            "以下暫時以原地展開的方式編輯："
        )
        _render_mask_editor_body()


def render_active_result_panel():
    """
    共用的「顯示分析結果 ＋ 編輯遮罩 ＋ 儲存」區塊。
    依 st.session_state.active_batch_pending_id / active_history_edit_index 決定
    按下「儲存」時的行為：一般新增一筆 / 從待分析清單移除後新增 / 更新既有的歷史紀錄列。
    呼叫端只需確保 current_mask / current_image_bgr / current_pixels_per_cm / last_result
    都已經設定好，並且在恰當的頁面位置（且只在合適的情境下）呼叫這個函式，避免同一個結果
    在頁面上重複出現。
    """
    if st.session_state.current_mask is None or st.session_state.last_result is None:
        return

    computed = recompute_full_result(
        st.session_state.current_mask,
        st.session_state.current_pixels_per_cm,
        st.session_state.height_threshold_cm,
    )
    render_result_visuals(st.session_state.current_image_bgr, st.session_state.current_mask, computed)
    st.caption(f"ArUco 編號比對：{st.session_state.last_result.get('編號比對結果', '')}")

    # ---------------- 筆刷編輯區塊 ----------------
    st.divider()
    edit_col1, edit_col2 = st.columns([1, 4])
    with edit_col1:
        if not st.session_state.editing_mode:
            if st.button("✏️ 編輯遮罩", use_container_width=True, key="edit_mask_btn"):
                st.session_state.mask_before_editing = st.session_state.current_mask.copy()
                st.session_state.rect_first_corner = None
                st.session_state.rect_second_corner = None
                st.session_state.last_click_coords = None
                st.session_state.editing_mode = True
                st.rerun()
        else:
            st.caption("編輯中…（請在彈出視窗中操作）")

    if st.session_state.editing_mode:
        mask_editor_dialog()

    # ---------------- 儲存按鈕（編輯中時隱藏，避免存到編輯前的舊資料） ----------------
    if not st.session_state.editing_mode:
        st.divider()
        is_history_edit = st.session_state.active_history_edit_index is not None
        is_batch = st.session_state.active_batch_pending_id is not None
        if is_history_edit:
            save_label, save_caption = "💾 更新這筆歷史紀錄", "會覆蓋原本這一筆的分析結果與照片，其他欄位維持不變。"
        elif is_batch:
            save_label, save_caption = "💾 儲存這筆結果並從待分析清單移除", "會寫入資料庫，並把這一筆從批次匯入的待分析清單移除。"
        else:
            save_label, save_caption = "💾 儲存此點位分析結果", "點擊左側按鈕，將目前顯示的分析結果與照片寫入下方多點位資料庫，並自動存成本機檔案。"

        save_col1, save_col2 = st.columns([1, 4])
        with save_col1:
            if st.button(save_label, type="primary", use_container_width=True, key="save_result_btn"):
                point_id_for_photo = st.session_state.last_result.get("點位編號", "point")
                ts = st.session_state.last_result.get("紀錄時間", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                photo_filename = save_photo_file(st.session_state.current_image_bgr, point_id_for_photo, ts)
                st.session_state.last_result["照片檔名"] = photo_filename

                if is_history_edit:
                    idx = st.session_state.active_history_edit_index
                    for k, v in st.session_state.last_result.items():
                        if k in st.session_state.dataframe.columns:
                            st.session_state.dataframe.at[idx, k] = v
                    save_local_data(st.session_state.dataframe)
                    st.session_state.active_history_edit_index = None
                    success_msg = "已更新這筆歷史紀錄！"
                elif is_batch:
                    new_row = pd.DataFrame([st.session_state.last_result])
                    st.session_state.dataframe = pd.concat(
                        [st.session_state.dataframe, new_row], ignore_index=True
                    )
                    save_local_data(st.session_state.dataframe)
                    st.session_state.pending_records = [
                        r for r in st.session_state.pending_records
                        if r["_pending_id"] != st.session_state.active_batch_pending_id
                    ]
                    st.session_state.active_batch_pending_id = None
                    success_msg = "已儲存並從待分析清單移除！"
                else:
                    new_row = pd.DataFrame([st.session_state.last_result])
                    st.session_state.dataframe = pd.concat(
                        [st.session_state.dataframe, new_row], ignore_index=True
                    )
                    save_local_data(st.session_state.dataframe)
                    success_msg = "已儲存至資料庫（含本機檔案備份）！"

                st.session_state.last_result = None
                st.session_state.current_mask = None
                st.session_state.current_image_bgr = None
                st.success(success_msg)
                st.rerun()
        with save_col2:
            st.caption(save_caption)


# ----------------------------------------------------------------------------
# 主程式
# ----------------------------------------------------------------------------
def main():
    init_session_state()

    st.title("🌿 街道非正式綠化（盆栽佔用）現場調查分析系統")
    st.caption("電腦視覺（ArUco 比例尺校正 + 植生遮罩擷取）× 環境行為學空間數據分析")

    # ==========================================================
    # 側邊欄：現場照片上傳 + 點位屬性表單（可隱藏）+ 影像分析參數（常駐）
    # ==========================================================
    with st.sidebar:
        toggle_label = "🙈 隱藏左側點位表單" if st.session_state.show_manual_form else "👁️ 顯示左側點位表單（單筆手動輸入用）"
        if st.button(toggle_label, use_container_width=True):
            st.session_state.show_manual_form = not st.session_state.show_manual_form
            st.rerun()

        if st.session_state.show_manual_form:
            st.header("📷 現場照片上傳")
            uploaded_file = st.file_uploader("上傳單張現場調查照片（JPG / PNG）", type=["jpg", "jpeg", "png"])

            ids_found = []
            if uploaded_file is not None:
                try:
                    preview_bytes = uploaded_file.getvalue()
                    preview_pil = Image.open(io.BytesIO(preview_bytes))
                    preview_bgr = pil_to_bgr(preview_pil)
                    ids_found, _dict_used_preview = detect_aruco_ids_only(preview_bgr)
                    st.image(bgr_to_rgb_for_display(preview_bgr), caption="已上傳照片預覽", use_container_width=True)
                except Exception:
                    st.error("照片讀取失敗，請確認檔案格式")

            if uploaded_file is not None:
                if ids_found:
                    st.success(f"📷 已偵測到 ArUco，ID：{'、'.join(str(i) for i in ids_found)}")
                else:
                    st.warning("📷 未偵測到 ArUco 標記，請確認畫面中是否有清楚拍到")

            with st.expander("📏 名詞說明（拍照距離 / 占用深度 / ArUco 擺放位置）"):
                st.markdown(
                    "- **拍照距離**：相機鏡頭到「建築外牆」的距離。\n"
                    "- **盆栽占用深度**：盆栽最外側（含枝條葉子）到「建築外牆」的距離。\n"
                    "- **ArUco 擺放位置**：請貼放在該點位「盆栽最外側」處，做為比例尺換算的基準，"
                    "確保面積／高度換算與占用深度的量測基準一致。"
                )

            st.divider()
            st.caption("以下欄位填寫時不會立即重新整理，全部填好後按最下方「執行影像分析」才會一次讀取。")

            with st.form("point_attribute_form", clear_on_submit=False):
                st.header("📋 點位屬性輸入")

                mrt_station = st.text_input("捷運站", placeholder="請手動輸入，例如：忠孝復興站")
                point_id = st.text_input("點位編號（Point ID）", value="")

                shooting_distance = st.number_input(
                    "拍照距離（公尺，以 0.5m 為單位）",
                    min_value=0.5, max_value=50.0, value=3.0, step=0.5, format="%.1f",
                    help="相機鏡頭到「建築外牆」的距離。",
                )

                location_type_raw = st.selectbox("擺放位置", LOCATION_OPTIONS)
                location_type_other = st.text_input(
                    "若擺放位置選「其他」，請在這裡填寫", key="location_other",
                    help="僅在上方「擺放位置」選擇『其他』時才會使用這裡填寫的內容。",
                )

                arrangement_type = st.selectbox("盆栽擺放型態", ARRANGEMENT_OPTIONS)

                st.markdown("**現場實測最高高度（公分）— 依擺放方式分別填寫**")
                h1, h2, h3 = st.columns(3)
                with h1:
                    floor_height_cm = st.number_input("落地擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="floor_h")
                with h2:
                    hanging_height_cm = st.number_input("吊掛擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="hang_h")
                with h3:
                    upward_height_cm = st.number_input("向上擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="up_h")

                occupancy_depth_cm = st.number_input(
                    "盆栽占用深度（公分）", min_value=0.0, max_value=500.0, value=0.0, step=1.0,
                    help="盆栽最外側（含枝條葉子）到「建築外牆」的距離。",
                )

                st.markdown("**盆栽擺放數量（盆）**")
                q1, q2, q3, q4 = st.columns(4)
                with q1:
                    floor_qty = st.number_input("落地", min_value=0, value=0, step=1, key="floor_q")
                with q2:
                    hanging_qty = st.number_input("吊掛", min_value=0, value=0, step=1, key="hang_q")
                with q3:
                    upward_qty = st.number_input("向上", min_value=0, value=0, step=1, key="up_q")
                with q4:
                    empty_qty = st.number_input("空盆栽", min_value=0, value=0, step=1, key="empty_q")

                st.caption("（數量合計會在按下「執行影像分析」後，於下方結果中顯示）")

                note_text = st.text_area("附註（選填）", value="", placeholder="其他需要記錄的現場觀察...")

                marker_real_length_cm = st.number_input(
                    "ArUco 實際邊長（公分）", min_value=1.0, max_value=100.0, value=20.0, step=0.5,
                    help="ArUco 標記本身的實際邊長，貼放於該點位盆栽最外側，作為比例尺校正基準。",
                )

                run_analysis = st.form_submit_button("🚀 執行影像分析", use_container_width=True, type="primary")

            if location_type_raw == "其他":
                location_type_final = location_type_other.strip() if location_type_other.strip() else "其他（未填寫）"
            else:
                location_type_final = location_type_raw

            total_qty_excl_empty = floor_qty + hanging_qty + upward_qty
        else:
            st.info(
                "已隱藏「照片上傳／點位屬性表單」（單筆手動輸入用）。"
                "如果你要用主畫面下方的「批次匯入」功能，不需要叫出這裡；"
                "有需要手動輸入單筆資料時，再按上方按鈕叫出來。"
            )
            uploaded_file = None
            ids_found = []
            mrt_station = ""
            point_id = ""
            shooting_distance = 3.0
            location_type_final = LOCATION_OPTIONS[0]
            arrangement_type = ARRANGEMENT_OPTIONS[0]
            floor_height_cm = hanging_height_cm = upward_height_cm = 0.0
            occupancy_depth_cm = 0.0
            floor_qty = hanging_qty = upward_qty = empty_qty = 0
            total_qty_excl_empty = 0
            note_text = ""
            marker_real_length_cm = 20.0
            run_analysis = False

        st.divider()
        st.header("🎨 影像分析參數")
        st.caption("這裡的設定不受上面表單顯示/隱藏影響，單筆分析與批次分析都會套用。")

        green_method = st.radio(
            "綠色遮罩演算法", ["HSV 色彩閾值", "Excess Green Index (ExG)"],
            help="決定用哪一種方式從照片中判斷「哪些像素是植栽的綠色」。HSV 用色相/飽和度/明度範圍篩選；ExG 用增強綠色的指數，對光線變化較不敏感。",
        )

        h_low = st.slider("Hue 下限（僅 HSV 模式使用）", 0, 179, 35, help="色相（顏色種類）篩選範圍的下限，數值越低越偏黃綠。")
        h_high = st.slider("Hue 上限（僅 HSV 模式使用）", 0, 179, 85, help="色相篩選範圍的上限，數值越高越偏藍綠。")
        s_low = st.slider("Saturation 下限（僅 HSV 模式使用）", 0, 255, 40, help="飽和度下限，數值越低會連同較淡、較灰的顏色也判定為植栽。")
        v_low = st.slider("Value 下限（僅 HSV 模式使用）", 0, 255, 40, help="明度下限，數值越低會連同較暗的陰影植栽也判定為植栽。")
        exg_threshold = st.slider("ExG 二值化閾值（僅 ExG 模式使用）", 0, 255, 130, help="數值越高，判定為植栽的門檻越嚴格，抓到的綠色範圍會變小。")

        morph_kernel = st.slider(
            "形態學去雜訊核大小", 1, 15, 5, step=2,
            help="用來去除遮罩上的小雜點、填補小空洞的運算範圍，數值越大平滑效果越強，但也可能抹掉細小的植栽枝葉。",
        )
        height_threshold_cm = st.slider(
            "高低型態判定閾值（公分）", 1.0, 20.0, HEIGHT_LEVEL_THRESHOLD_DEFAULT, step=0.5,
            help="左/中/右三區高度要相差多少公分以上，才會被視為有明顯的「高、低」差異，藉此判斷擺放立面形式。",
        )

        st.divider()
        manual_scale_override = st.number_input(
            "手動輸入比例尺（像素/公分）— 僅於 ArUco 未偵測到時使用，0 表示不啟用",
            min_value=0.0, value=0.0, step=0.1,
            help="當照片中沒有清楚偵測到 ArUco 標記時，可以自己估算「畫面中多少像素等於 1 公分」，手動輸入來替代自動校正。",
        )

    # ==========================================================
    # 主畫面：影像分析流程（初次分析）
    # ==========================================================
    if run_analysis:
        if uploaded_file is None:
            st.error("請先於左側面板上傳一張現場調查照片。")
        else:
            pil_image = Image.open(uploaded_file)
            image_bgr = pil_to_bgr(pil_image)

            annotated_img, pixels_per_cm, marker_center, dict_used, marker_corners, marker_id = detect_aruco_marker(
                image_bgr, marker_real_length_cm
            )

            scale_source = None
            if pixels_per_cm is not None:
                scale_source = f"ArUco 自動偵測（字典：{dict_used}，ID：{marker_id}）"
                st.success(f"✅ 成功偵測到 ArUco Marker（{dict_used}，ID={marker_id}），比例尺 = {pixels_per_cm:.3f} px/cm")
            else:
                st.warning("⚠️ 未偵測到任何 ArUco Marker，請於側邊欄手動輸入預估比例尺（像素/公分）。")
                if manual_scale_override and manual_scale_override > 0:
                    pixels_per_cm = manual_scale_override
                    scale_source = "手動輸入"
                    st.info(f"目前採用手動比例尺：{pixels_per_cm:.3f} px/cm")
                else:
                    scale_source = "未提供（無法換算實際面積/高度）"

            if green_method == "HSV 色彩閾值":
                raw_mask = extract_green_mask_hsv(image_bgr, h_low, h_high, s_low, v_low)
            else:
                raw_mask = extract_green_mask_exg(image_bgr, exg_threshold)

            mask = clean_mask(raw_mask, kernel_size=morph_kernel)

            match_msg, match_status = build_aruco_match_message(point_id, ids_found)

            # 將本次分析的原圖 / 遮罩 / 比例尺 / 判定閾值存入 session_state，
            # 供筆刷編輯與重新計算使用（閾值滑桿在表單內，僅在送出當下的值會生效）
            st.session_state.current_image_bgr = image_bgr
            st.session_state.current_mask = mask
            st.session_state.current_pixels_per_cm = pixels_per_cm
            st.session_state.height_threshold_cm = height_threshold_cm
            st.session_state.editing_mode = False
            st.session_state.canvas_key_counter += 1

            computed = recompute_full_result(mask, pixels_per_cm, height_threshold_cm)

            st.session_state.last_result = {
                "捷運站": mrt_station,
                "點位編號": point_id if point_id else f"P-{datetime.now().strftime('%H%M%S')}",
                "ArUco偵測ID": "、".join(str(i) for i in ids_found) if ids_found else "",
                "編號比對結果": match_status,
                "拍照距離(m)": shooting_distance,
                "擺放位置": location_type_final,
                "盆栽擺放型態": arrangement_type,
                "落地擺放高度(cm)": floor_height_cm,
                "吊掛擺放高度(cm)": hanging_height_cm,
                "向上擺放高度(cm)": upward_height_cm,
                "盆栽占用深度(cm)": occupancy_depth_cm,
                "落地擺放數量": floor_qty,
                "吊掛擺放數量": hanging_qty,
                "向上擺放數量": upward_qty,
                "空盆栽數量": empty_qty,
                "擺放數量合計(不含空盆)": total_qty_excl_empty,
                "附註": note_text,
                "ArUco邊長(cm)": marker_real_length_cm,
                "比例尺來源": scale_source,
                "像素/公分比例尺": round(pixels_per_cm, 4) if pixels_per_cm else None,
                "AI辨識綠化面積(m2)": round(computed["area_m2"], 4) if computed["area_m2"] is not None else None,
                "左區高度(cm)": computed["zone_heights_cm"].get("left"),
                "中區高度(cm)": computed["zone_heights_cm"].get("mid"),
                "右區高度(cm)": computed["zone_heights_cm"].get("right"),
                "高低型態判定": computed["shape_label"],
                "照片檔名": "",
                "紀錄時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    # ==========================================================
    # 分析結果顯示（單筆手動輸入這條路徑；批次／歷史編輯有各自的顯示位置）
    # ==========================================================
    if st.session_state.active_batch_pending_id is None and st.session_state.active_history_edit_index is None:
        render_active_result_panel()

    # ==========================================================
    # 批次匯入點位屬性（Excel）＋ 逐筆上傳照片並執行分析
    # ==========================================================
    st.divider()
    st.header("📥 批次匯入點位屬性（Excel）")
    st.caption(
        "可以先在 Excel 範本裡一次填好幾十筆點位的基本資料，匯入後**不會馬上分析**；"
        "之後針對每一筆資料上傳對應照片，再按「執行分析」逐筆處理、逐筆確認結果後再儲存。"
    )

    tmpl_col1, tmpl_col2 = st.columns(2)
    with tmpl_col1:
        st.download_button(
            "⬇️ 下載 Excel 範本",
            data=build_template_excel_bytes(),
            file_name="街道非正式綠化_點位資料範本.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with tmpl_col2:
        uploaded_template = st.file_uploader("上傳填好的 Excel 檔案", type=["xlsx"], key="template_uploader")
        if uploaded_template is not None:
            if st.button("📥 加入待分析清單", use_container_width=True):
                new_records, tmpl_error = parse_template_excel(uploaded_template.getvalue())
                if tmpl_error:
                    st.error(tmpl_error)
                else:
                    for rec in new_records:
                        st.session_state.pending_id_counter += 1
                        rec["_pending_id"] = st.session_state.pending_id_counter
                        st.session_state.pending_records.append(rec)
                    st.success(f"已加入 {len(new_records)} 筆到待分析清單。")
                    st.rerun()

    if st.session_state.pending_records:
        st.subheader(f"🕒 待分析清單（共 {len(st.session_state.pending_records)} 筆）")
        pending_display_df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "_pending_id"} for r in st.session_state.pending_records]
        )
        st.dataframe(pending_display_df, use_container_width=True)

        st.markdown("**📸 批次上傳照片（依照片中的 ArUco 編號自動比對清單中的資料）**")
        st.caption(
            "一次選取多張照片，程式會讀出每張照片裡的 ArUco 編號，"
            "找出待分析清單中「ArUco編號」欄位相同的那一筆，自動完成分析並直接存入資料庫"
            "（不會逐筆停下來讓你確認）。之後如果某幾筆想要微調遮罩，"
            "到下方「🔍 歷史紀錄查詢」裡叫出照片編輯即可。"
        )
        bulk_photos = st.file_uploader(
            "選取多張照片", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk_photo_uploader"
        )
        if bulk_photos:
            if st.button("🚀 批次比對並分析所有照片", type="primary", key="bulk_analyze_btn"):
                matched_count, no_match_count, no_aruco_count = 0, 0, 0
                results_log = []

                for f in bulk_photos:
                    try:
                        bulk_pil = Image.open(f)
                        bulk_bgr = pil_to_bgr(bulk_pil)
                    except Exception:
                        results_log.append(f"❌ {f.name}：照片讀取失敗")
                        continue

                    bulk_ids_found, _ = detect_aruco_ids_only(bulk_bgr)
                    if not bulk_ids_found:
                        no_aruco_count += 1
                        results_log.append(f"❌ {f.name}：未偵測到 ArUco 標記")
                        continue

                    matched_record = None
                    for rec in st.session_state.pending_records:
                        try:
                            rec_aruco_id = int(float(rec.get("ArUco編號", -1)))
                        except (TypeError, ValueError):
                            continue
                        if rec_aruco_id in bulk_ids_found:
                            matched_record = rec
                            break

                    if matched_record is None:
                        no_match_count += 1
                        results_log.append(
                            f"⚠️ {f.name}：偵測到 ArUco ID {bulk_ids_found}，"
                            "但待分析清單中找不到「ArUco編號」相符的資料（請確認 Excel 裡有沒有填對）"
                        )
                        continue

                    marker_len_b = float(matched_record.get("ArUco邊長(cm)", 20) or 20)
                    _ann_b, pxcm_b, _c, dict_used_b, _mc, marker_id_b = detect_aruco_marker(bulk_bgr, marker_len_b)
                    if pxcm_b is not None:
                        scale_source_b = f"ArUco 自動偵測（字典：{dict_used_b}，ID：{marker_id_b}）"
                    elif manual_scale_override and manual_scale_override > 0:
                        pxcm_b = manual_scale_override
                        scale_source_b = "手動輸入"
                    else:
                        scale_source_b = "未提供（無法換算實際面積/高度）"

                    if green_method == "HSV 色彩閾值":
                        raw_mask_b = extract_green_mask_hsv(bulk_bgr, h_low, h_high, s_low, v_low)
                    else:
                        raw_mask_b = extract_green_mask_exg(bulk_bgr, exg_threshold)
                    mask_b = clean_mask(raw_mask_b, kernel_size=morph_kernel)
                    computed_b = recompute_full_result(mask_b, pxcm_b, height_threshold_cm)
                    match_msg_b, match_status_b = build_aruco_match_message(
                        matched_record.get("點位編號", ""), bulk_ids_found
                    )

                    floor_qty_b = int(float(matched_record.get("落地擺放數量", 0) or 0))
                    hanging_qty_b = int(float(matched_record.get("吊掛擺放數量", 0) or 0))
                    upward_qty_b = int(float(matched_record.get("向上擺放數量", 0) or 0))
                    empty_qty_b = int(float(matched_record.get("空盆栽數量", 0) or 0))
                    ts_b = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    photo_filename_b = save_photo_file(bulk_bgr, matched_record.get("點位編號", ""), ts_b)

                    row_b = {
                        "捷運站": matched_record.get("捷運站", ""),
                        "點位編號": matched_record.get("點位編號", ""),
                        "ArUco編號": matched_record.get("ArUco編號", ""),
                        "ArUco偵測ID": "、".join(str(i) for i in bulk_ids_found),
                        "編號比對結果": match_status_b,
                        "拍照距離(m)": matched_record.get("拍照距離(m)", 0),
                        "擺放位置": matched_record.get("擺放位置", ""),
                        "盆栽擺放型態": matched_record.get("盆栽擺放型態", ""),
                        "落地擺放高度(cm)": matched_record.get("落地擺放高度(cm)", 0),
                        "吊掛擺放高度(cm)": matched_record.get("吊掛擺放高度(cm)", 0),
                        "向上擺放高度(cm)": matched_record.get("向上擺放高度(cm)", 0),
                        "盆栽占用深度(cm)": matched_record.get("盆栽占用深度(cm)", 0),
                        "落地擺放數量": floor_qty_b,
                        "吊掛擺放數量": hanging_qty_b,
                        "向上擺放數量": upward_qty_b,
                        "空盆栽數量": empty_qty_b,
                        "擺放數量合計(不含空盆)": floor_qty_b + hanging_qty_b + upward_qty_b,
                        "附註": matched_record.get("附註", ""),
                        "ArUco邊長(cm)": marker_len_b,
                        "比例尺來源": scale_source_b,
                        "像素/公分比例尺": round(pxcm_b, 4) if pxcm_b else None,
                        "AI辨識綠化面積(m2)": round(computed_b["area_m2"], 4) if computed_b["area_m2"] is not None else None,
                        "左區高度(cm)": computed_b["zone_heights_cm"].get("left"),
                        "中區高度(cm)": computed_b["zone_heights_cm"].get("mid"),
                        "右區高度(cm)": computed_b["zone_heights_cm"].get("right"),
                        "高低型態判定": computed_b["shape_label"],
                        "照片檔名": photo_filename_b,
                        "紀錄時間": ts_b,
                    }
                    st.session_state.dataframe = pd.concat(
                        [st.session_state.dataframe, pd.DataFrame([row_b])], ignore_index=True
                    )
                    st.session_state.pending_records = [
                        r for r in st.session_state.pending_records if r["_pending_id"] != matched_record["_pending_id"]
                    ]
                    matched_count += 1
                    results_log.append(f"✅ {f.name}：已比對到點位「{matched_record.get('點位編號', '')}」，完成分析並儲存")

                save_local_data(st.session_state.dataframe)
                st.success(
                    f"批次處理完成：成功 {matched_count} 筆、找不到對應資料 {no_match_count} 筆、"
                    f"未偵測到 ArUco {no_aruco_count} 筆。"
                )
                with st.expander("查看詳細比對紀錄"):
                    for line in results_log:
                        st.write(line)
                st.rerun()

        st.divider()
        st.markdown("**🖐️ 或者，逐筆手動上傳照片並確認結果**")

        pending_options = {r["_pending_id"]: r["點位編號"] for r in st.session_state.pending_records}
        selected_pending_id = st.selectbox(
            "選擇要處理的點位",
            options=list(pending_options.keys()),
            format_func=lambda pid: f"{pending_options[pid]}（清單序號 {pid}）",
            key="pending_select",
        )
        selected_record = next(r for r in st.session_state.pending_records if r["_pending_id"] == selected_pending_id)

        with st.expander("這筆的屬性內容", expanded=False):
            detail_series = pd.Series({k: v for k, v in selected_record.items() if k != "_pending_id"})
            st.table(detail_series.rename("值"))

        remove_col, _ = st.columns([1, 4])
        with remove_col:
            if st.button("🗑️ 從清單移除這一筆", key=f"remove_pending_{selected_pending_id}"):
                st.session_state.pending_records = [
                    r for r in st.session_state.pending_records if r["_pending_id"] != selected_pending_id
                ]
                st.rerun()

        pending_photo = st.file_uploader(
            "為這筆點位上傳照片", type=["jpg", "jpeg", "png"], key=f"pending_photo_{selected_pending_id}"
        )

        if pending_photo is not None:
            pending_pil = Image.open(pending_photo)
            pending_bgr = pil_to_bgr(pending_pil)
            pending_ids_found, _ = detect_aruco_ids_only(pending_bgr)
            st.image(bgr_to_rgb_for_display(pending_bgr), caption="已上傳照片預覽", use_container_width=True)
            if pending_ids_found:
                st.success(f"📷 已偵測到 ArUco，ID：{'、'.join(str(i) for i in pending_ids_found)}")
            else:
                st.warning("📷 未偵測到 ArUco 標記")

            if st.button("🚀 對這筆資料執行分析", type="primary", key=f"analyze_pending_{selected_pending_id}"):
                marker_len = float(selected_record.get("ArUco邊長(cm)", 20) or 20)
                _annotated_p, pixels_per_cm_p, _c, dict_used_p, _mc, marker_id_p = detect_aruco_marker(pending_bgr, marker_len)

                if pixels_per_cm_p is not None:
                    scale_source_p = f"ArUco 自動偵測（字典：{dict_used_p}，ID：{marker_id_p}）"
                elif manual_scale_override and manual_scale_override > 0:
                    pixels_per_cm_p = manual_scale_override
                    scale_source_p = "手動輸入"
                else:
                    scale_source_p = "未提供（無法換算實際面積/高度）"

                if green_method == "HSV 色彩閾值":
                    raw_mask_p = extract_green_mask_hsv(pending_bgr, h_low, h_high, s_low, v_low)
                else:
                    raw_mask_p = extract_green_mask_exg(pending_bgr, exg_threshold)
                mask_p = clean_mask(raw_mask_p, kernel_size=morph_kernel)

                computed_p = recompute_full_result(mask_p, pixels_per_cm_p, height_threshold_cm)
                match_msg_p, match_status_p = build_aruco_match_message(selected_record.get("點位編號", ""), pending_ids_found)

                floor_qty_p = int(float(selected_record.get("落地擺放數量", 0) or 0))
                hanging_qty_p = int(float(selected_record.get("吊掛擺放數量", 0) or 0))
                upward_qty_p = int(float(selected_record.get("向上擺放數量", 0) or 0))
                empty_qty_p = int(float(selected_record.get("空盆栽數量", 0) or 0))

                # 統一寫入 current_* 共用狀態，讓這筆結果可以使用跟單筆輸入相同的
                # 「顯示結果／編輯遮罩／儲存」流程（含彈出視窗筆刷編輯）
                st.session_state.current_image_bgr = pending_bgr
                st.session_state.current_mask = mask_p
                st.session_state.current_pixels_per_cm = pixels_per_cm_p
                st.session_state.height_threshold_cm = height_threshold_cm
                st.session_state.editing_mode = False
                st.session_state.canvas_key_counter += 1
                st.session_state.active_batch_pending_id = selected_pending_id
                st.session_state.active_history_edit_index = None
                st.session_state.last_result = {
                    "捷運站": selected_record.get("捷運站", ""),
                    "點位編號": selected_record.get("點位編號", ""),
                    "ArUco編號": selected_record.get("ArUco編號", ""),
                    "ArUco偵測ID": "、".join(str(i) for i in pending_ids_found) if pending_ids_found else "",
                    "編號比對結果": match_status_p,
                    "拍照距離(m)": selected_record.get("拍照距離(m)", 0),
                    "擺放位置": selected_record.get("擺放位置", ""),
                    "盆栽擺放型態": selected_record.get("盆栽擺放型態", ""),
                    "落地擺放高度(cm)": selected_record.get("落地擺放高度(cm)", 0),
                    "吊掛擺放高度(cm)": selected_record.get("吊掛擺放高度(cm)", 0),
                    "向上擺放高度(cm)": selected_record.get("向上擺放高度(cm)", 0),
                    "盆栽占用深度(cm)": selected_record.get("盆栽占用深度(cm)", 0),
                    "落地擺放數量": floor_qty_p,
                    "吊掛擺放數量": hanging_qty_p,
                    "向上擺放數量": upward_qty_p,
                    "空盆栽數量": empty_qty_p,
                    "擺放數量合計(不含空盆)": floor_qty_p + hanging_qty_p + upward_qty_p,
                    "附註": selected_record.get("附註", ""),
                    "ArUco邊長(cm)": marker_len,
                    "比例尺來源": scale_source_p,
                    "像素/公分比例尺": round(pixels_per_cm_p, 4) if pixels_per_cm_p else None,
                    "AI辨識綠化面積(m2)": round(computed_p["area_m2"], 4) if computed_p["area_m2"] is not None else None,
                    "左區高度(cm)": computed_p["zone_heights_cm"].get("left"),
                    "中區高度(cm)": computed_p["zone_heights_cm"].get("mid"),
                    "右區高度(cm)": computed_p["zone_heights_cm"].get("right"),
                    "高低型態判定": computed_p["shape_label"],
                    "照片檔名": "",
                    "紀錄時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.rerun()

        # 顯示「目前選定這一筆」的分析結果（跟單筆輸入共用同一套顯示／編輯遮罩／儲存流程）
        if st.session_state.active_batch_pending_id == selected_pending_id:
            render_active_result_panel()

    # ==========================================================
    # 歷史紀錄查詢：依捷運站篩選 / 排序 / 點選查看單筆詳細內容（含照片）
    # ==========================================================
    df = st.session_state.dataframe

    st.divider()
    st.header("🔍 歷史紀錄查詢")

    if df.empty:
        st.info("目前尚無已儲存的點位資料。請於上方完成單筆分析或批次匯入後儲存。")
    else:
        df_view = df.copy()
        df_view["紀錄日期"] = pd.to_datetime(df_view["紀錄時間"], errors="coerce").dt.date.astype(str)

        station_values = sorted(
            v for v in df_view["捷運站"].astype(str).str.strip().unique() if v and v.lower() != "nan"
        )
        station_options = ["全部"] + station_values

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            selected_station = st.selectbox("篩選捷運站", station_options, key="station_filter")
        with q_col2:
            sort_field = st.selectbox("排序依據", ["紀錄時間", "捷運站", "點位編號"])
        with q_col3:
            sort_order = st.radio("排序方式", ["新到舊／Z→A", "舊到新／A→Z"], horizontal=True)

        if selected_station != "全部":
            df_view = df_view[df_view["捷運站"].astype(str).str.strip() == selected_station]

        ascending = sort_order.startswith("舊到新")
        try:
            df_view = df_view.sort_values(by=sort_field, ascending=ascending)
        except Exception:
            pass

        st.caption(f"目前顯示：{selected_station}，共 {len(df_view)} 筆")
        st.dataframe(df_view.drop(columns=["紀錄日期"]), use_container_width=True)

        st.subheader("📌 查看單一點位詳細內容")
        if not df_view.empty:
            # 用「點位編號｜捷運站｜紀錄時間」組成顯示用標籤，並對應到資料表原始的 index，
            # 避免不同批次匯入時點位編號重複，導致選到別筆卻顯示同一張照片。
            label_for_index = {}
            for idx, row in df_view.iterrows():
                base_label = f"{row.get('點位編號', '')}｜{row.get('捷運站', '')}｜{row.get('紀錄時間', '')}"
                label = base_label
                dup_n = 1
                while label in label_for_index:
                    dup_n += 1
                    label = f"{base_label}（重複 #{dup_n}）"
                label_for_index[label] = idx

            selected_label = st.selectbox(
                "選擇要查看的點位（點位編號｜捷運站｜紀錄時間）",
                list(label_for_index.keys()),
                key="history_select",
            )
            record_index = label_for_index[selected_label]
            record = df_view.loc[record_index]
            selected_point = record.get("點位編號", "")

            detail_col1, detail_col2 = st.columns([1, 1])
            with detail_col1:
                st.markdown(f"**捷運站**：{record.get('捷運站', '')}")
                st.markdown(f"**擺放位置**：{record.get('擺放位置', '')}")
                st.markdown(f"**盆栽擺放型態**：{record.get('盆栽擺放型態', '')}")
                st.markdown(f"**擺放形式判定**：{record.get('高低型態判定', '')}")
                st.markdown(f"**AI辨識綠化面積**：{record.get('AI辨識綠化面積(m2)', '')} m²")
                st.markdown(f"**盆栽數量合計（不含空盆）**：{record.get('擺放數量合計(不含空盆)', '')}")
                st.markdown(f"**附註**：{record.get('附註', '')}")
                st.markdown(f"**紀錄時間**：{record.get('紀錄時間', '')}")
            with detail_col2:
                photo_filename = record.get("照片檔名", "")
                photo_path = os.path.join(PHOTOS_DIR, str(photo_filename)) if photo_filename else ""
                if photo_filename and os.path.exists(photo_path):
                    st.image(photo_path, caption="現場照片", use_container_width=True)
                else:
                    st.caption("（此筆紀錄沒有可顯示的照片）")

                st.markdown("**重新上傳這筆的照片**")
                replacement_photo = st.file_uploader(
                    "選擇新照片取代目前的照片", type=["jpg", "jpeg", "png"],
                    key=f"replace_photo_{record_index}",
                )
                if replacement_photo is not None:
                    if st.button("🔄 更新這筆的照片", key=f"replace_photo_btn_{record_index}"):
                        new_pil = Image.open(replacement_photo)
                        new_bgr = pil_to_bgr(new_pil)
                        new_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_filename = save_photo_file(new_bgr, selected_point, new_ts)
                        st.session_state.dataframe.at[record_index, "照片檔名"] = new_filename
                        save_local_data(st.session_state.dataframe)
                        st.success("照片已更新！")
                        st.rerun()

                if photo_filename and os.path.exists(photo_path):
                    if st.button("✏️ 載入這張照片以編輯遮罩", key=f"reload_edit_{record_index}"):
                        loaded_bgr = imread_unicode(photo_path)
                        if loaded_bgr is None:
                            st.error(
                                "讀取照片失敗（可能是檔案損毀，或路徑中含有中文字元造成讀取問題）。"
                                "可以試試看用左邊「重新上傳這筆的照片」換一張。"
                            )
                        else:
                            if green_method == "HSV 色彩閾值":
                                raw_mask_h = extract_green_mask_hsv(loaded_bgr, h_low, h_high, s_low, v_low)
                            else:
                                raw_mask_h = extract_green_mask_exg(loaded_bgr, exg_threshold)
                            mask_h = clean_mask(raw_mask_h, kernel_size=morph_kernel)

                            # 比例尺優先沿用這筆紀錄原本存的像素/公分比例尺；沒有的話才重新偵測 ArUco
                            stored_scale = None
                            raw_scale_val = record.get("像素/公分比例尺", None)
                            try:
                                if raw_scale_val not in (None, "", "None") and not pd.isna(raw_scale_val):
                                    stored_scale = float(raw_scale_val)
                            except (TypeError, ValueError):
                                stored_scale = None
                            if stored_scale is None:
                                marker_len_h = float(record.get("ArUco邊長(cm)", 20) or 20)
                                _ann_h, stored_scale, _c, _d, _mc, _mid = detect_aruco_marker(loaded_bgr, marker_len_h)

                            st.session_state.current_image_bgr = loaded_bgr
                            st.session_state.current_mask = mask_h
                            st.session_state.current_pixels_per_cm = stored_scale
                            st.session_state.height_threshold_cm = height_threshold_cm
                            st.session_state.last_result = {col: record.get(col, "") for col in DATAFRAME_COLUMNS}
                            st.session_state.active_history_edit_index = record_index
                            st.session_state.active_batch_pending_id = None
                            st.session_state.editing_mode = False
                            st.rerun()
                    st.caption("用目前側邊欄的影像分析參數重新產生遮罩（不一定跟當初存檔時完全相同），可以在下方微調後更新這筆紀錄。")

            if st.session_state.active_history_edit_index == record_index:
                render_active_result_panel()

        # ==========================================================
        # 統計分析區塊
        # ==========================================================
        st.divider()
        st.header("📈 統計分析")

        st.subheader("連續變數描述性統計")
        numeric_df = df[NUMERIC_COLS_FOR_STATS].apply(pd.to_numeric, errors="coerce")
        if numeric_df.dropna(how="all").empty:
            st.info("尚無足夠的數值資料可供統計。")
        else:
            stats_table = pd.DataFrame({
                "平均值 (Mean)": numeric_df.mean(),
                "中位數 (Median)": numeric_df.median(),
                "標準差 (Std Dev)": numeric_df.std(),
                "最大值 (Max)": numeric_df.max(),
                "最小值 (Min)": numeric_df.min(),
                "樣本數 (N)": numeric_df.count(),
            }).round(3)
            st.dataframe(stats_table, use_container_width=True)

        st.subheader("類別變數次數分配")
        cat_cols = st.columns(len(CATEGORICAL_COLS_FOR_STATS))
        for i, col_name in enumerate(CATEGORICAL_COLS_FOR_STATS):
            with cat_cols[i]:
                counts = df[col_name].value_counts()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(counts.index.astype(str), counts.values, color="#4CAF50")
                ax.set_title(col_name, fontsize=10)
                ax.set_ylabel("次數")
                plt.xticks(rotation=30, ha="right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ==========================================================
    # 多點位資料庫：完整備份／還原 ＋ 可直接編輯的原始資料表（移到最下方，
    # 平常瀏覽建議用上方的「歷史紀錄查詢」，這裡保留給需要大量編輯欄位或備份/還原的情境）
    # ==========================================================
    st.divider()
    st.header("🗄️ 多點位資料庫")
    st.caption(f"資料自動存於本機檔案：{DATA_FILE_PATH}")

    st.subheader("📦 完整備份與還原（含照片，可跨電腦使用）")
    st.caption(
        "匯出會把「所有已儲存的資料」與「所有現場照片」打包成一個 ZIP 檔案。"
        "只要有這個 ZIP 檔，不管在哪一台電腦（Windows／MacBook／雲端版）都能匯入還原成一樣的狀態，"
        "很適合當作正式備份，或是把資料從一台電腦搬到另一台。"
    )
    backup_col1, backup_col2 = st.columns(2)
    with backup_col1:
        st.markdown("**⬇️ 匯出備份**")
        backup_zip_bytes = build_full_backup_zip(st.session_state.dataframe)
        st.download_button(
            "📦 匯出完整備份（ZIP，含照片）",
            data=backup_zip_bytes,
            file_name=f"street_greening_full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with backup_col2:
        st.markdown("**⬆️ 匯入還原**")
        uploaded_backup = st.file_uploader("選擇備份 ZIP 檔案", type=["zip"], key="backup_zip_uploader")
        restore_mode_label = st.radio(
            "還原方式",
            ["取代目前所有資料", "合併（附加到目前資料後面）"],
            horizontal=True,
            key="restore_mode",
        )
        restore_mode = "replace" if restore_mode_label.startswith("取代") else "merge"
        if restore_mode == "replace":
            st.caption("⚠️ 這會清空目前畫面上的資料庫，改成備份檔裡的內容，請確認後再按下方按鈕。")
        if uploaded_backup is not None:
            if st.button("🔁 還原此備份", type="primary", use_container_width=True):
                success, message = restore_from_backup_zip(uploaded_backup.getvalue(), mode=restore_mode)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    reload_col, _ = st.columns([1, 4])
    with reload_col:
        if st.button("🔄 重新讀取本機資料檔（例如剛從雲端硬碟同步完成時使用）"):
            st.session_state.dataframe = load_local_data()
            st.rerun()

    df = st.session_state.dataframe

    if df.empty:
        st.info("目前尚無已儲存的點位資料。請於上方完成分析後點擊「儲存此點位分析結果」。")
    else:
        st.caption("下表可直接點擊儲存格編輯內容；將滑鼠移到某一列最左側可勾選並按刪除鍵（🗑）移除整列。")

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="db_editor",
        )

        if not edited_df.equals(df):
            st.session_state.dataframe = edited_df.reset_index(drop=True)
            save_local_data(st.session_state.dataframe)
            df = st.session_state.dataframe

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                "⬇️ 匯出為 CSV（備份）",
                data=df_to_csv_bytes(df),
                file_name=f"street_greening_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "⬇️ 匯出為 Excel（備份）",
                data=df_to_excel_bytes(df),
                file_name=f"street_greening_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )



if __name__ == "__main__":
    main()
