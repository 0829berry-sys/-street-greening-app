# -*- coding: utf-8 -*-
"""
==================================================================================
街道非正式綠化（盆栽佔用）現場調查分析系統  v3
Streamlit + OpenCV 電腦視覺應用程式
==================================================================================

【安裝相依套件】
    pip install streamlit opencv-python numpy pandas pillow matplotlib openpyxl xlsxwriter streamlit-drawable-canvas

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
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

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

CANVAS_MAX_WIDTH = 640  # 編輯畫布最大寬度（像素），避免畫面過大


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
    filename = f"{safe_id}_{timestamp_str.replace(':', '').replace(' ', '_').replace('-', '')}.jpg"
    filepath = os.path.join(PHOTOS_DIR, filename)
    try:
        cv2.imwrite(filepath, image_bgr)
        return filename
    except Exception:
        return ""


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
    if "height_threshold_cm" not in st.session_state:
        st.session_state.height_threshold_cm = HEIGHT_LEVEL_THRESHOLD_DEFAULT


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
    point_id_clean = (point_id or "").strip()

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


def resize_for_canvas(image_bgr, max_width=CANVAS_MAX_WIDTH):
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr.copy()
    scale = max_width / w
    resized = cv2.resize(image_bgr, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized


def apply_canvas_strokes_to_mask(mask, canvas_image_data):
    """
    將畫布上的筆刷筆跡（RGBA，小尺寸）套用回原始尺寸的遮罩：
    綠色筆跡（新增）→ 該處遮罩設為 255；紅色筆跡（移除）→ 該處遮罩設為 0。
    """
    if canvas_image_data is None:
        return mask

    h_orig, w_orig = mask.shape[:2]
    canvas_rgb = canvas_image_data[:, :, :3].astype(np.uint8)
    alpha = canvas_image_data[:, :, 3]

    canvas_rgb_full = cv2.resize(canvas_rgb, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    alpha_full = cv2.resize(alpha, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    drawn = alpha_full > 10
    r_ch, g_ch, b_ch = canvas_rgb_full[:, :, 0], canvas_rgb_full[:, :, 1], canvas_rgb_full[:, :, 2]

    is_add_stroke = drawn & (g_ch > 150) & (r_ch < 100) & (b_ch < 100)
    is_erase_stroke = drawn & (r_ch > 150) & (g_ch < 100) & (b_ch < 100)

    new_mask = mask.copy()
    new_mask[is_add_stroke] = 255
    new_mask[is_erase_stroke] = 0
    return new_mask


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
# 主程式
# ----------------------------------------------------------------------------
def main():
    init_session_state()

    st.title("🌿 街道非正式綠化（盆栽佔用）現場調查分析系統")
    st.caption("電腦視覺（ArUco 比例尺校正 + 植生遮罩擷取）× 環境行為學空間數據分析")

    # ==========================================================
    # 側邊欄：現場照片上傳（移到最上方）+ 點位屬性表單 + 影像分析參數
    # ==========================================================
    with st.sidebar:
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

        with st.expander("📏 名詞說明（拍照距離 / 占用深度 / ArUco 擺放位置）"):
            st.markdown(
                "- **拍照距離**：相機鏡頭到「建築外牆」的距離。\n"
                "- **盆栽占用深度**：盆栽最外側（含枝條葉子）到「建築外牆」的距離。\n"
                "- **ArUco 擺放位置**：請貼放在該點位「盆栽最外側」處，做為比例尺換算的基準，"
                "確保面積／高度換算與占用深度的量測基準一致。"
            )

        st.divider()
        st.header("📋 點位屬性輸入")

        mrt_station = st.text_input("捷運站", placeholder="請手動輸入，例如：忠孝復興站")

        point_id = st.text_input("點位編號（Point ID）", value="")
        if uploaded_file is not None:
            match_msg, match_status = build_aruco_match_message(point_id, ids_found)
            if match_status == "相符":
                st.success(f"📷 {match_msg}")
            elif match_status == "不符":
                st.warning(f"📷 {match_msg}")
            elif match_status == "未偵測到":
                st.warning(f"📷 {match_msg}，請確認畫面中是否有清楚拍到 ArUco 標記")
            else:
                st.info(f"📷 {match_msg}")
        else:
            match_msg, match_status = "尚未上傳照片，無法自動比對 ArUco 編號", "未上傳"
            st.caption(f"📷 {match_msg}")

        shooting_distance = st.number_input(
            "拍照距離（公尺，以 0.5m 為單位）",
            min_value=0.5, max_value=50.0, value=3.0, step=0.5, format="%.1f",
            help="相機鏡頭到「建築外牆」的距離。",
        )

        location_type_raw = st.selectbox("擺放位置", LOCATION_OPTIONS)
        if location_type_raw == "其他":
            location_type_other = st.text_input("請手動輸入擺放位置（其他）", key="location_other")
            location_type_final = location_type_other.strip() if location_type_other.strip() else "其他（未填寫）"
        else:
            location_type_final = location_type_raw

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

        total_qty_excl_empty = floor_qty + hanging_qty + upward_qty
        st.info(f"➕ 盆栽數量合計（不含空盆）：**{total_qty_excl_empty}** 盆　｜　空盆栽數量：**{empty_qty}** 盆")

        note_text = st.text_area("附註（選填）", value="", placeholder="其他需要記錄的現場觀察...")

        marker_real_length_cm = st.number_input(
            "ArUco 實際邊長（公分）", min_value=1.0, max_value=100.0, value=20.0, step=0.5,
            help="ArUco 標記本身的實際邊長，貼放於該點位盆栽最外側，作為比例尺校正基準。",
        )

        st.divider()
        st.header("🎨 影像分析參數")

        green_method = st.radio(
            "綠色遮罩演算法", ["HSV 色彩閾值", "Excess Green Index (ExG)"],
            help="決定用哪一種方式從照片中判斷「哪些像素是植栽的綠色」。HSV 用色相/飽和度/明度範圍篩選；ExG 用增強綠色的指數，對光線變化較不敏感。",
        )

        if green_method == "HSV 色彩閾值":
            h_low = st.slider("Hue 下限", 0, 179, 35, help="色相（顏色種類）篩選範圍的下限，數值越低越偏黃綠。")
            h_high = st.slider("Hue 上限", 0, 179, 85, help="色相篩選範圍的上限，數值越高越偏藍綠。")
            s_low = st.slider("Saturation 下限", 0, 255, 40, help="飽和度下限，數值越低會連同較淡、較灰的顏色也判定為植栽。")
            v_low = st.slider("Value 下限", 0, 255, 40, help="明度下限，數值越低會連同較暗的陰影植栽也判定為植栽。")
            exg_threshold = None
        else:
            exg_threshold = st.slider("ExG 二值化閾值", 0, 255, 130, help="數值越高，判定為植栽的門檻越嚴格，抓到的綠色範圍會變小。")
            h_low = h_high = s_low = v_low = None

        morph_kernel = st.slider(
            "形態學去雜訊核大小", 1, 15, 5, step=2,
            help="用來去除遮罩上的小雜點、填補小空洞的運算範圍，數值越大平滑效果越強，但也可能抹掉細小的植栽枝葉。",
        )
        height_threshold_cm = st.slider(
            "高低型態判定閾值（公分）", 1.0, 20.0, st.session_state.height_threshold_cm, step=0.5,
            help="左/中/右三區高度要相差多少公分以上，才會被視為有明顯的「高、低」差異，藉此判斷擺放立面形式。",
        )
        st.session_state.height_threshold_cm = height_threshold_cm

        st.divider()
        manual_scale_override = st.number_input(
            "手動輸入比例尺（像素/公分）— 僅於 ArUco 未偵測到時使用，0 表示不啟用",
            min_value=0.0, value=0.0, step=0.1,
            help="當照片中沒有清楚偵測到 ArUco 標記時，可以自己估算「畫面中多少像素等於 1 公分」，手動輸入來替代自動校正。",
        )

        run_analysis = st.button("🚀 執行影像分析", use_container_width=True, type="primary")

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

            # 將本次分析的原圖 / 遮罩 / 比例尺存入 session_state，供筆刷編輯與重新計算使用
            st.session_state.current_image_bgr = image_bgr
            st.session_state.current_mask = mask
            st.session_state.current_pixels_per_cm = pixels_per_cm
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
            st.session_state._point_id_for_save = point_id

    # ==========================================================
    # 分析結果顯示（每次 rerun 都依 session_state 目前的 mask 重新畫）
    # ==========================================================
    if st.session_state.current_mask is not None and st.session_state.last_result is not None:
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
                if st.button("✏️ 編輯遮罩", use_container_width=True):
                    st.session_state.editing_mode = True
                    st.rerun()
            else:
                st.caption("編輯中…")

        if st.session_state.editing_mode:
            if not CANVAS_AVAILABLE:
                st.error(
                    "尚未安裝筆刷編輯所需套件，請於終端機執行："
                    "pip install streamlit-drawable-canvas 後重新啟動程式。"
                )
                if st.button("結束編輯（返回原始結果）"):
                    st.session_state.editing_mode = False
                    st.rerun()
            else:
                st.info(
                    "🖌️ 用滑鼠在下方畫布上塗抹：**綠色筆刷＝新增到綠化面積**、"
                    "**紅色筆刷＝從綠化面積移除**。塗抹完按「🔄 重新計算」套用並預覽，"
                    "滿意後按「✅ 完成編輯」才會固定最終結果並可以儲存。"
                )
                brush_mode = st.radio("筆刷模式", ["新增（綠色）", "移除（紅色）"], horizontal=True)
                brush_size = st.slider("筆刷大小", 5, 60, 20)
                stroke_color = "#00FF00" if brush_mode.startswith("新增") else "#FF0000"

                preview_bgr = resize_for_canvas(st.session_state.current_image_bgr)
                canvas_bg = Image.fromarray(bgr_to_rgb_for_display(preview_bgr))

                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=brush_size,
                    stroke_color=stroke_color,
                    background_image=canvas_bg,
                    update_streamlit=True,
                    height=preview_bgr.shape[0],
                    width=preview_bgr.shape[1],
                    drawing_mode="freedraw",
                    key=f"mask_editor_canvas_{st.session_state.canvas_key_counter}",
                )

                recalc_col, finish_col = st.columns(2)
                with recalc_col:
                    if st.button("🔄 重新計算（套用目前筆刷）", use_container_width=True):
                        if canvas_result is not None and canvas_result.image_data is not None:
                            st.session_state.current_mask = apply_canvas_strokes_to_mask(
                                st.session_state.current_mask, canvas_result.image_data
                            )
                            st.session_state.canvas_key_counter += 1  # 清空畫布，避免重複套用
                        st.rerun()
                with finish_col:
                    if st.button("✅ 完成編輯", type="primary", use_container_width=True):
                        if canvas_result is not None and canvas_result.image_data is not None:
                            st.session_state.current_mask = apply_canvas_strokes_to_mask(
                                st.session_state.current_mask, canvas_result.image_data
                            )
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
                        st.session_state.canvas_key_counter += 1
                        st.success("已完成編輯，最終結果已更新，可以儲存了。")
                        st.rerun()

        # ---------------- 儲存按鈕（編輯中時隱藏，避免存到編輯前的舊資料） ----------------
        if not st.session_state.editing_mode:
            st.divider()
            save_col1, save_col2 = st.columns([1, 4])
            with save_col1:
                if st.button("💾 儲存此點位分析結果", type="primary", use_container_width=True):
                    point_id_for_photo = st.session_state.last_result.get("點位編號", "point")
                    ts = st.session_state.last_result.get("紀錄時間", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    photo_filename = save_photo_file(st.session_state.current_image_bgr, point_id_for_photo, ts)
                    st.session_state.last_result["照片檔名"] = photo_filename

                    new_row = pd.DataFrame([st.session_state.last_result])
                    st.session_state.dataframe = pd.concat(
                        [st.session_state.dataframe, new_row], ignore_index=True
                    )
                    save_local_data(st.session_state.dataframe)

                    st.session_state.last_result = None
                    st.session_state.current_mask = None
                    st.session_state.current_image_bgr = None
                    st.success("已儲存至資料庫（含本機檔案備份）！")
                    st.rerun()
            with save_col2:
                st.caption("點擊左側按鈕，將目前顯示的分析結果與照片寫入下方多點位資料庫，並自動存成本機檔案。")

    # ==========================================================
    # 資料庫檢視 / 編輯 / 刪除 / 匯出
    # ==========================================================
    st.divider()
    st.header("🗄️ 多點位資料庫")
    st.caption(f"資料自動存於本機檔案：{DATA_FILE_PATH}")

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

        # ==========================================================
        # 歷史紀錄查詢：排序 / 分組 / 點選查看單筆詳細內容（含照片）
        # ==========================================================
        st.divider()
        st.header("🔍 歷史紀錄查詢")

        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            group_by_field = st.selectbox("分組依據", ["不分組", "捷運站", "紀錄日期"])
        with q_col2:
            sort_field = st.selectbox("排序依據", ["紀錄時間", "捷運站", "點位編號"])
        with q_col3:
            sort_order = st.radio("排序方式", ["新到舊／Z→A", "舊到新／A→Z"], horizontal=True)

        df_view = df.copy()
        df_view["紀錄日期"] = pd.to_datetime(df_view["紀錄時間"], errors="coerce").dt.date.astype(str)
        ascending = sort_order.startswith("舊到新")
        try:
            df_view = df_view.sort_values(by=sort_field, ascending=ascending)
        except Exception:
            pass

        if group_by_field == "不分組":
            st.dataframe(df_view.drop(columns=["紀錄日期"]), use_container_width=True)
        else:
            for group_name, group_df in df_view.groupby(group_by_field):
                with st.expander(f"{group_by_field}：{group_name}（{len(group_df)} 筆）"):
                    st.dataframe(group_df.drop(columns=["紀錄日期"]), use_container_width=True)

        st.subheader("📌 查看單一點位詳細內容")
        point_options = df_view["點位編號"].tolist()
        if point_options:
            selected_point = st.selectbox("選擇要查看的點位編號", point_options, key="history_select")
            record_rows = df_view[df_view["點位編號"] == selected_point]
            record = record_rows.iloc[-1]

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


if __name__ == "__main__":
    main()
