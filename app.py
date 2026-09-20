# -*- coding: utf-8 -*-
"""
==================================================================================
街道非正式綠化（盆栽佔用）現場調查分析系統  v2
Streamlit + OpenCV 電腦視覺應用程式
==================================================================================

【安裝相依套件】
    pip install streamlit opencv-python numpy pandas pillow matplotlib openpyxl xlsxwriter

【啟動方式】
    streamlit run app.py
    （若出現 'streamlit' 無法辨識，改用：python -m streamlit run app.py）

【v2 更新重點】
    1. 點位屬性表單新增：捷運站、盆栽占用深度、盆栽擺放數量（分落地/吊掛/向上/空盆並自動加總）、
       盆栽擺放位置（相對於人行道/建物）、附註；擺放位置與盆栽擺放位置皆可選「其他」手動輸入。
    2. 點位編號改為手動輸入，並自動偵測上傳照片中的 ArUco ID，於欄位下方即時提示是否偵測到、
       以及是否與手動編號相符。
    3. 拍照距離改以 0.5 公尺為單位。
    4. 盆栽擺放型態選項調整為：橫向擺放／直列擺放／群聚擺放／單盆擺放。
    5. 現場實測最高高度拆分為：落地擺放／吊掛擺放／向上擺放，三者可分別填入。
    6. 資料庫改用可編輯表格（st.data_editor），支援直接編輯儲存格、刪除整列，並保留
       CSV / Excel 匯出備份功能。
==================================================================================
"""

import io
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------------
# 全域設定
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="街道非正式綠化（盆栽佔用）調查分析系統",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATAFRAME_COLUMNS = [
    "捷運站",
    "點位編號",
    "ArUco偵測ID",
    "編號比對結果",
    "拍照距離(m)",
    "擺放位置",
    "盆栽擺放型態",
    "盆栽主要類型",
    "落地擺放高度(cm)",
    "吊掛擺放高度(cm)",
    "向上擺放高度(cm)",
    "盆栽占用深度(cm)",
    "落地擺放數量",
    "吊掛擺放數量",
    "向上擺放數量",
    "空盆栽數量",
    "擺放數量合計(不含空盆)",
    "盆栽擺放位置(相對位置)",
    "附註",
    "ArUco邊長(cm)",
    "比例尺來源",
    "像素/公分比例尺",
    "AI辨識綠化面積(m2)",
    "左區高度(cm)",
    "中區高度(cm)",
    "右區高度(cm)",
    "高低型態判定",
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

CATEGORICAL_COLS_FOR_STATS = ["擺放位置", "盆栽擺放型態", "盆栽擺放位置(相對位置)", "高低型態判定"]

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
RELATIVE_POSITION_OPTIONS = ["靠牆", "退縮", "佔滿通行寬度", "其他"]


# ----------------------------------------------------------------------------
# Session State 初始化
# ----------------------------------------------------------------------------
def init_session_state():
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


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
            # 取偵測到的第一個 marker 做比例尺校正
            marker_corners = corners[0].reshape(4, 2)  # shape (4,2): 左上,右上,右下,左下
            marker_id = int(ids.flatten()[0])

            # 計算四邊長度並取平均，作為 marker 邊長（像素）
            side_lengths_px = [
                np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
                for i in range(4)
            ]
            avg_side_px = float(np.mean(side_lengths_px))

            pixels_per_cm = avg_side_px / marker_real_length_cm

            # 繪製所有偵測到的 marker 邊界框
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            # 繪製第一個 marker 的幾何中心
            center_xy = marker_corners.mean(axis=0)
            cv2.circle(
                annotated,
                (int(center_xy[0]), int(center_xy[1])),
                6,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                annotated,
                f"{dict_name} ID:{marker_id} | {avg_side_px:.1f}px = {marker_real_length_cm}cm",
                (int(marker_corners[0][0]), max(int(marker_corners[0][1]) - 15, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            return annotated, pixels_per_cm, center_xy, dict_name, marker_corners, marker_id

    # 全部字典皆未偵測到
    return annotated, None, None, None, None, None


# ----------------------------------------------------------------------------
# 綠色植栽遮罩擷取
# ----------------------------------------------------------------------------
def extract_green_mask_hsv(image_bgr, h_low, h_high, s_low, v_low):
    """HSV 色彩空間之綠色閾值遮罩"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def extract_green_mask_exg(image_bgr, exg_threshold):
    """Excess Green Index (ExG = 2G - R - B) 植生遮罩"""
    img = image_bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, exg_threshold, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask, kernel_size=5):
    """形態學開閉運算，去除雜訊、填補空洞"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean


def compute_green_area_m2(mask, pixels_per_cm):
    """依比例尺將遮罩像素數換算為實際面積（平方公尺）"""
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
    """
    掃描遮罩每一欄（column），找出最頂端（Y 最小）之綠色像素位置（Upper Envelope）。
    將有效欄位範圍等分為左/中/右三區，計算各區平均頂端高度（相對於畫面底部，換算為公分）。

    回傳：
        zone_heights_cm: dict {"left":.., "mid":.., "right":..}（可能含 None）
        envelope_points: list of (x, y_top)，供繪圖
        x_bounds: (x_min, x_split1, x_split2, x_max)
    """
    h, w = mask.shape[:2]
    col_has_green = np.any(mask > 0, axis=0)
    valid_x = np.where(col_has_green)[0]

    if len(valid_x) == 0:
        return {"left": None, "mid": None, "right": None}, [], None

    x_min, x_max = int(valid_x.min()), int(valid_x.max())
    span = x_max - x_min
    x_split1 = x_min + span // 3
    x_split2 = x_min + 2 * span // 3

    # 逐欄找最頂端 y（第一個非零像素的 row index）
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
        height_px = h - avg_y_top  # 以畫面底部視為地面基準
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
    依左/中/右三區高度關係，自動分類擺放立面形式：
        - 齊平型：三區高度落差皆小於閾值
        - 凹型（高低高）：中間明顯低於左右兩側
        - 拱型（低高低）：中間明顯高於左右兩側
        - 遞減型（高高低）：由左至右明顯遞減
        - 遞增型（低高高）：由左至右明顯遞增
        - 不規則型：其餘無法歸類的組合
    """
    hl, hm, hr = zone_heights_cm.get("left"), zone_heights_cm.get("mid"), zone_heights_cm.get("right")
    if hl is None or hm is None or hr is None:
        return "資料不足（無法判定）"

    vals = [hl, hm, hr]
    if max(vals) - min(vals) < threshold_cm:
        return "齊平型"

    if (hl - hm) > threshold_cm and (hr - hm) > threshold_cm:
        return "凹型（高低高）"

    if (hm - hl) > threshold_cm and (hm - hr) > threshold_cm:
        return "拱型（低高低）"

    if (hl - hm) > threshold_cm * 0.5 and (hm - hr) > threshold_cm * 0.5 and (hl - hr) > threshold_cm:
        return "遞減型（高高低／左高右低）"

    if (hm - hl) > threshold_cm * 0.5 and (hr - hm) > threshold_cm * 0.5 and (hr - hl) > threshold_cm:
        return "遞增型（低高高／左低右高）"

    if (hl - hr) > threshold_cm:
        return "遞減型（高高低／左高右低）"
    if (hr - hl) > threshold_cm:
        return "遞增型（低高高／左低右高）"

    return "不規則型"


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

    # 畫上緣輪廓線（Upper Envelope）
    pts = np.array(sorted(envelope_points, key=lambda p: p[0]), dtype=np.int32)
    for i in range(len(pts) - 1):
        cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 140, 255), 2)

    # 畫左中右分區垂直分隔線
    cv2.line(vis, (x_split1, 0), (x_split1, h), (255, 255, 0), 2)
    cv2.line(vis, (x_split2, 0), (x_split2, h), (255, 255, 0), 2)

    # 標註三區高度數值
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
# 主程式
# ----------------------------------------------------------------------------
def main():
    init_session_state()

    st.title("🌿 街道非正式綠化（盆栽佔用）現場調查分析系統")
    st.caption("電腦視覺（ArUco 比例尺校正 + 植生遮罩擷取）× 環境行為學空間數據分析")

    # ==========================================================
    # 側邊欄：點位屬性表單 + 影像分析參數
    # ==========================================================
    with st.sidebar:
        st.header("📋 點位屬性輸入")

        # 1. 捷運站（手動填答，無下拉選項）
        mrt_station = st.text_input("捷運站", placeholder="請手動輸入，例如：忠孝復興站")

        # 2. 點位編號（手動輸入）+ ArUco 偵測狀態提示（placeholder，稍後於上傳照片後填入）
        point_id = st.text_input("點位編號（Point ID）", value="")
        aruco_check_placeholder = st.empty()
        aruco_check_placeholder.caption("📷 尚未上傳照片，無法自動比對 ArUco 編號")

        # 3. 拍照距離（0.5m 為單位）
        shooting_distance = st.number_input(
            "拍照距離（公尺，以 0.5m 為單位）",
            min_value=0.5, max_value=50.0, value=3.0, step=0.5, format="%.1f"
        )

        # 4. 擺放位置（新增騎樓、其他）
        location_type_raw = st.selectbox("擺放位置", LOCATION_OPTIONS)
        if location_type_raw == "其他":
            location_type_other = st.text_input("請手動輸入擺放位置（其他）", key="location_other")
            location_type_final = location_type_other.strip() if location_type_other.strip() else "其他（未填寫）"
        else:
            location_type_final = location_type_raw

        # 5. 盆栽擺放型態（調整選項）
        arrangement_type = st.selectbox("盆栽擺放型態", ARRANGEMENT_OPTIONS)

        # 盆栽主要類型（保留原欄位）
        plant_type = st.text_input("盆栽主要類型", placeholder="例如：觀葉植物、木本灌木、多肉草花")

        st.markdown("**現場實測最高高度（公分）— 依擺放方式分別填寫**")
        h1, h2, h3 = st.columns(3)
        with h1:
            floor_height_cm = st.number_input("落地擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="floor_h")
        with h2:
            hanging_height_cm = st.number_input("吊掛擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="hang_h")
        with h3:
            upward_height_cm = st.number_input("向上擺放", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="up_h")

        # 7. 盆栽占用深度
        occupancy_depth_cm = st.number_input("盆栽占用深度（公分）", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

        # 8. 盆栽擺放數量（分落地/吊掛/向上/空盆）
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

        # 9. 即時加總（不含空盆栽，空盆栽另外顯示）
        total_qty_excl_empty = floor_qty + hanging_qty + upward_qty
        st.info(f"➕ 盆栽數量合計（不含空盆）：**{total_qty_excl_empty}** 盆　｜　空盆栽數量：**{empty_qty}** 盆")

        # 10. 盆栽擺放位置（相對於人行道/建物）
        rel_position_raw = st.selectbox("盆栽擺放位置（相對於人行道／建物）", RELATIVE_POSITION_OPTIONS)
        if rel_position_raw == "其他":
            rel_position_other = st.text_input("請手動輸入盆栽擺放位置（其他）", key="rel_pos_other")
            rel_position_final = rel_position_other.strip() if rel_position_other.strip() else "其他（未填寫）"
        else:
            rel_position_final = rel_position_raw

        # 11. 附註（選填）
        note_text = st.text_area("附註（選填）", value="", placeholder="其他需要記錄的現場觀察...")

        # ArUco 實際邊長（比例尺校正用）
        marker_real_length_cm = st.number_input("ArUco 實際邊長（公分）", min_value=1.0, max_value=100.0, value=20.0, step=0.5)

        st.divider()
        st.header("🎨 影像分析參數")

        green_method = st.radio("綠色遮罩演算法", ["HSV 色彩閾值", "Excess Green Index (ExG)"])

        if green_method == "HSV 色彩閾值":
            h_low = st.slider("Hue 下限", 0, 179, 35)
            h_high = st.slider("Hue 上限", 0, 179, 85)
            s_low = st.slider("Saturation 下限", 0, 255, 40)
            v_low = st.slider("Value 下限", 0, 255, 40)
            exg_threshold = None
        else:
            exg_threshold = st.slider("ExG 二值化閾值", 0, 255, 130)
            h_low = h_high = s_low = v_low = None

        morph_kernel = st.slider("形態學去雜訊核大小", 1, 15, 5, step=2)
        height_threshold_cm = st.slider("高低型態判定閾值（公分）", 1.0, 20.0, HEIGHT_LEVEL_THRESHOLD_DEFAULT, step=0.5)

        st.divider()
        st.header("📷 現場照片上傳")
        uploaded_file = st.file_uploader("上傳單張現場調查照片（JPG / PNG）", type=["jpg", "jpeg", "png"])

        # ---- 上傳後立即進行 ArUco ID 快速偵測，回填點位編號下方的提示 ----
        ids_found = []
        if uploaded_file is not None:
            try:
                preview_bytes = uploaded_file.getvalue()
                preview_pil = Image.open(io.BytesIO(preview_bytes))
                preview_bgr = pil_to_bgr(preview_pil)
                ids_found, _dict_used_preview = detect_aruco_ids_only(preview_bgr)
                match_msg, match_status = build_aruco_match_message(point_id, ids_found)
                if match_status == "相符":
                    aruco_check_placeholder.success(f"📷 {match_msg}")
                elif match_status == "不符":
                    aruco_check_placeholder.warning(f"📷 {match_msg}")
                elif match_status == "未偵測到":
                    aruco_check_placeholder.warning(f"📷 {match_msg}，請確認畫面中是否有清楚拍到 ArUco 標記")
                else:
                    aruco_check_placeholder.info(f"📷 {match_msg}")
            except Exception:
                aruco_check_placeholder.error("📷 照片讀取失敗，無法進行 ArUco 偵測")
                match_msg, match_status = "照片讀取失敗", "無法比對"
        else:
            match_msg, match_status = "尚未上傳照片，無法自動比對 ArUco 編號", "未上傳"

        st.divider()
        manual_scale_override = st.number_input(
            "手動輸入比例尺（像素/公分）— 僅於 ArUco 未偵測到時使用，0 表示不啟用",
            min_value=0.0, value=0.0, step=0.1
        )

        run_analysis = st.button("🚀 執行影像分析", use_container_width=True, type="primary")

    # ==========================================================
    # 主畫面：影像分析流程
    # ==========================================================
    if run_analysis:
        if uploaded_file is None:
            st.error("請先於左側面板上傳一張現場調查照片。")
        else:
            pil_image = Image.open(uploaded_file)
            image_bgr = pil_to_bgr(pil_image)

            # -------- Step 1：ArUco 偵測與比例尺校正 --------
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

            # -------- Step 2：綠色植栽遮罩擷取 --------
            if green_method == "HSV 色彩閾值":
                raw_mask = extract_green_mask_hsv(image_bgr, h_low, h_high, s_low, v_low)
            else:
                raw_mask = extract_green_mask_exg(image_bgr, exg_threshold)

            mask = clean_mask(raw_mask, kernel_size=morph_kernel)
            area_m2, green_pixel_count = compute_green_area_m2(mask, pixels_per_cm)

            # -------- Step 3：立面輪廓左/中/右高度分析 --------
            zone_heights_cm, envelope_points, x_bounds = analyze_zone_profile(mask, pixels_per_cm)
            shape_label = classify_zone_shape(zone_heights_cm, threshold_cm=height_threshold_cm)

            # -------- Step 4：視覺化三聯圖 --------
            mask_overlay = image_bgr.copy()
            mask_overlay[mask > 0] = (0, 255, 0)
            mask_blend = cv2.addWeighted(image_bgr, 0.5, mask_overlay, 0.5, 0)

            envelope_vis = draw_envelope_visualization(image_bgr, envelope_points, x_bounds, zone_heights_cm)

            st.subheader("🖼️ 影像分析視覺化比對")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(bgr_to_rgb_for_display(annotated_img), caption="原圖 + ArUco 偵測", use_container_width=True)
            with col2:
                st.image(bgr_to_rgb_for_display(mask_blend), caption="綠色植栽遮罩疊圖", use_container_width=True)
            with col3:
                st.image(bgr_to_rgb_for_display(envelope_vis), caption="立面上緣輪廓 + 左中右分區", use_container_width=True)

            # -------- Step 5：分析結果摘要 --------
            st.subheader("📊 本次分析結果摘要")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("比例尺 (px/cm)", f"{pixels_per_cm:.2f}" if pixels_per_cm else "N/A")
            m2.metric("AI 辨識綠化面積 (m²)", f"{area_m2:.4f}" if area_m2 is not None else "N/A")
            m3.metric("擺放形式判定", shape_label)
            m4.metric("綠色像素數", f"{green_pixel_count:,}")

            z1, z2, z3 = st.columns(3)
            z1.metric("左區高度 (cm)", zone_heights_cm.get("left") if zone_heights_cm.get("left") is not None else "N/A")
            z2.metric("中區高度 (cm)", zone_heights_cm.get("mid") if zone_heights_cm.get("mid") is not None else "N/A")
            z3.metric("右區高度 (cm)", zone_heights_cm.get("right") if zone_heights_cm.get("right") is not None else "N/A")

            st.caption(f"ArUco 編號比對：{match_msg}")

            # 暫存本次結果，供「儲存」按鈕使用
            st.session_state.last_result = {
                "捷運站": mrt_station,
                "點位編號": point_id if point_id else f"P-{datetime.now().strftime('%H%M%S')}",
                "ArUco偵測ID": "、".join(str(i) for i in ids_found) if ids_found else "",
                "編號比對結果": match_status,
                "拍照距離(m)": shooting_distance,
                "擺放位置": location_type_final,
                "盆栽擺放型態": arrangement_type,
                "盆栽主要類型": plant_type,
                "落地擺放高度(cm)": floor_height_cm,
                "吊掛擺放高度(cm)": hanging_height_cm,
                "向上擺放高度(cm)": upward_height_cm,
                "盆栽占用深度(cm)": occupancy_depth_cm,
                "落地擺放數量": floor_qty,
                "吊掛擺放數量": hanging_qty,
                "向上擺放數量": upward_qty,
                "空盆栽數量": empty_qty,
                "擺放數量合計(不含空盆)": total_qty_excl_empty,
                "盆栽擺放位置(相對位置)": rel_position_final,
                "附註": note_text,
                "ArUco邊長(cm)": marker_real_length_cm,
                "比例尺來源": scale_source,
                "像素/公分比例尺": round(pixels_per_cm, 4) if pixels_per_cm else None,
                "AI辨識綠化面積(m2)": round(area_m2, 4) if area_m2 is not None else None,
                "左區高度(cm)": zone_heights_cm.get("left"),
                "中區高度(cm)": zone_heights_cm.get("mid"),
                "右區高度(cm)": zone_heights_cm.get("right"),
                "高低型態判定": shape_label,
                "紀錄時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    # ==========================================================
    # 儲存目前分析結果至資料庫
    # ==========================================================
    if st.session_state.last_result is not None:
        st.divider()
        save_col1, save_col2 = st.columns([1, 4])
        with save_col1:
            if st.button("💾 儲存此點位分析結果", type="primary", use_container_width=True):
                new_row = pd.DataFrame([st.session_state.last_result])
                st.session_state.dataframe = pd.concat(
                    [st.session_state.dataframe, new_row], ignore_index=True
                )
                st.session_state.last_result = None
                st.success("已儲存至資料庫！")
                st.rerun()
        with save_col2:
            st.caption("點擊左側按鈕，將目前顯示的分析結果寫入下方多點位資料庫（每一筆上傳都會保留，不會互相覆蓋）。")

    # ==========================================================
    # 資料庫檢視 / 編輯 / 刪除 / 匯出
    # ==========================================================
    st.divider()
    st.header("🗄️ 多點位資料庫")

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

        # 將使用者於表格中做的編輯／刪除同步回資料庫
        if not edited_df.equals(df):
            st.session_state.dataframe = edited_df.reset_index(drop=True)
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
