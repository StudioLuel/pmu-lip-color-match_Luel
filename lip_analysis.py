# -*- coding: utf-8 -*-
"""
lip_analysis.py — 색채과학 코어 모듈
======================================
PMU Lip Color Match 학습 플랫폼의 색상 분석 엔진.

[이 모듈의 역할]
- 입술 사진에서 Lab 색상값 추출 (그림자 배제, 노이즈 제거, 화이트밸런스 보정)
- ΔE2000(인지 색차) 기준으로 "색상군(Yellow/Gray/Red × Brown/Purple)" 분류
- 분류 결과에 실무 자료(반영구 교재) 기반 톤업/중화 레시피 매칭
- 색소 배합 최적화 (ΔE2000 최소화)

[중요] 이 모듈이 내놓는 모든 결과는 "참고용 가설"입니다.
현재 이 프로그램의 목적은 "정답을 내려주는 도구"가 아니라
"초급 시술자가 스스로 판단하고, 그 판단을 검증해보는 학습 보조 도구"입니다.
따라서 모든 반환값에는 ΔE00(근거 수치)과 신뢰도(confidence)가 함께 따라붙습니다.
"""

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import color
from sklearn.cluster import KMeans

# =========================================================
# 0. 상수 / 색채과학 기준값
# =========================================================

# CIEDE2000 기준 "평균 관찰자가 겨우 인지 가능한 차이(JND)".
PERCEPTUAL_JND = 2.3

# 1·2위 분류 후보의 ΔE00 차이가 이보다 작으면 "경계선 케이스"로 간주.
CONFIDENCE_MARGIN = 1.5

# 마스크(입술 영역)가 전체 이미지 대비 이 비율보다 작으면 세그멘테이션 실패 경고.
MIN_LIP_AREA_RATIO = 0.03

# "하얀 입술(색소가 거의 없는 입술)" 판정 기준 — a*(붉은기)가 이보다 낮으면 해당.
WHITE_LIP_A_THRESHOLD = 10.0
WHITE_LIP_L_THRESHOLD = 60.0


# ---------------------------------------------------------
# 색소 데이터베이스
# ⚠️ 아래 Lab 값은 색상 설명(붉은기/채도/웜쿨 등)을 바탕으로 한 추정치입니다.
#    분광측색계 실측값이 아닙니다. 케이스 라이브러리에 실측 사진이 쌓이면
#    이 값들을 교체/보정해야 합니다.
#    색소명은 반영구 시술 교재(첨부자료)에서 실제 사용되는 Perma Lux / Perma Classic
#    라인업 명칭을 그대로 반영했습니다.
# ---------------------------------------------------------
PIGMENT_DB = {
    "Perma Lux": {
        "로즈로얄 (Rose Royale)": {"lab": [45.0, 38.0, 8.0], "status": "active",
                                   "note": "자연스러운 웜계열 레드. 베어의 대체 색상."},
        "진 (Gin)": {"lab": [32.0, 42.0, 10.0], "status": "active",
                      "note": "강한 중화/딥레드. 로얄레드의 대체 색상."},
        "라라 (Lara)": {"lab": [50.0, 40.0, -2.0], "status": "discontinued",
                         "note": "단종 — 루루스로즈로 대체.", "replacement": "루루스로즈 (Luluh Rose)"},
        "말로 (Mallow)": {"lab": [40.0, 35.0, 2.0], "status": "active",
                           "note": "쿨톤 레드/머루빛. amz s2의 대체 색상."},
        "포피 (Poppy)": {"lab": [48.0, 45.0, 18.0], "status": "active",
                          "note": "선명한 웜레드."},
        "조이 (Joy)": {"lab": [38.0, 40.0, 5.0], "status": "active",
                        "note": "로얄레드 단종 대체 색상."},
        "헨나 (Henna)": {"lab": [35.0, 20.0, 15.0], "status": "active",
                          "note": "브라운 계열. amz s1(헨나4+로즈로얄1)의 대체 성분."},
        "헤이즈 (Haze)": {"lab": [55.0, 5.0, 8.0], "status": "active",
                           "note": "톤다운/회색조 교정용 어저스터. 단독 사용 거의 없음."},
        "어드저스트 (Adjust)": {"lab": [70.0, 2.0, 2.0], "status": "active",
                                 "note": "믹싱 솔루션(색상 아님) — 농도 조절용."},
    },
    "Perma Classic": {
        "패션레드 (Passion Red, 포피 계열)": {"lab": [46.0, 44.0, 16.0], "status": "active", "note": "-"},
        "로얄레드 (Royal Red, 조이 계열)": {"lab": [38.0, 40.0, 5.0], "status": "discontinued",
                                            "note": "단종 — 조이로 대체.", "replacement": "조이 (Joy)"},
        "라즈베리 (Raspberry)": {"lab": [42.0, 48.0, -5.0], "status": "active",
                                  "note": "쿨톤 비비드 핑크."},
        "크림드핑크 (Creamed Pink)": {"lab": [62.0, 25.0, 6.0], "status": "active",
                                        "note": "대표 톤업 색상. 웜·쿨 모두 베이스로 사용."},
        "애플리콧 (Apricot)": {"lab": [58.0, 28.0, 22.0], "status": "active",
                                 "note": "웜톤 보정/톤업 보조 색상."},
    },
    "이븐플로": {
        "베어 (Bare)": {"lab": [44.0, 36.0, 6.0], "status": "discontinued",
                         "note": "단종 — 로즈로얄로 대체.", "replacement": "로즈로얄 (Rose Royale)"},
        "루루스로즈 (Luluh Rose)": {"lab": [50.0, 38.0, -3.0], "status": "active",
                                     "note": "라라 단종 대체 색상."},
    },
    "월드페이머스": {
        "막스 스킨톤#1 (Max Skintone #1)": {"lab": [50.0, 12.0, 18.0], "status": "active",
                                              "note": "멀멀립(자연 베이지) 배합용. amz s1 대체 성분(헨나4+로즈로얄1)으로도 활용."},
    },
}


def iter_active_pigments(pigment_db: dict) -> list:
    """활성(단종 아닌) 색소만 배합 탐색 대상으로 추출."""
    out = []
    for brand, pigs in pigment_db.items():
        for name, meta in pigs.items():
            if meta.get("status") == "discontinued":
                continue
            out.append({"brand": brand, "name": name, "lab": np.array(meta["lab"], dtype=float)})
    return out


def find_pigment_replacement(pigment_name: str, pigment_db: dict):
    """단종 색소명을 입력하면 대체 색소 정보를 반환."""
    for brand, pigs in pigment_db.items():
        for name, meta in pigs.items():
            if pigment_name.strip() in name and meta.get("status") == "discontinued":
                return {"brand": brand, "name": name, "replacement": meta.get("replacement", "정보 없음"),
                         "note": meta.get("note", "")}
    return None


# ---------------------------------------------------------
# 색상군 레퍼런스 (반영구 시술 교재 "다크립 중화" 표 기반)
# ---------------------------------------------------------
REFERENCE_TONES = {
    "healthy_warm": {
        "lab": [58.0, 35.0, 16.0], "healthy": True, "label_ko": "건강한 웜톤",
    },
    "healthy_cool": {
        "lab": [58.0, 32.0, 9.0], "healthy": True, "label_ko": "건강한 쿨톤 (선호 발색)",
    },
    "yellow_brown": {
        "lab": [45.0, 25.0, 22.0], "healthy": False,
        "label_ko": "Yellow Brown (노란기 강한 브라운)",
        "direction": "이미 웜한 입술이므로 옐로우를 추가하는 방식은 피하고, 웜한 브라운을 잡으면서 혈색을 살려줍니다.",
        "toneup": "크림드핑크",
        "neutralize_direction": "쿨톤핑크색",
        "recipe_mild": "말로6 + 진1",
        "recipe_severe": "말로3 + 크림드핑크1(생략가능) + 진1 + 루루스로즈1 전부믹스",
    },
    "gray_brown": {
        "lab": [42.0, 10.0, 10.0], "healthy": False,
        "label_ko": "Gray Brown (회기 강한 브라운)",
        "direction": "채도를 살리는 것이 가장 중요합니다. 너무 누디하거나 탁한 색을 더하면 회색기가 더 남아 보일 수 있습니다.",
        "toneup": "크림드핑크",
        "neutralize_direction": "채도 높은 핑크형광",
        "recipe_mild": "크림드핑크로 1차 가볍게, 그 위에 루루스로즈4+진1로 단독 중화",
        "recipe_severe": "루루스로즈4 + 진1 (심한 부분은 레이어링 추가)",
    },
    "gray_purple": {
        "lab": [40.0, 8.0, -5.0], "healthy": False,
        "label_ko": "Gray Purple (회기 강한 보라색)",
        "direction": "채도를 살리는 것이 가장 중요합니다. 담배를 많이 피우는 고객에게 자주 보이는 컬러입니다.",
        "toneup": "크림드핑크1 + 애플리콧1",
        "neutralize_direction": "채도 높은 코랄형광",
        "recipe_mild": "크핑+애플리콧으로 1차 가볍게, 그 위에 진으로 단독 중화",
        "recipe_severe": "진 단독",
    },
    "red_brown": {
        "lab": [35.0, 32.0, 14.0], "healthy": False,
        "label_ko": "Red Brown (검붉은 브라운)",
        "direction": "명도를 올리면서 동시에 검붉은 색을 따뜻한 방향으로 이동시킵니다. 오렌지 컬러가 과하면 레드브라운이 더 강조될 수 있으니 주의하세요.",
        "toneup": "크림드핑크3 + 애플리콧1",
        "neutralize_direction": "쿨핑크 + 살몬컬러 + 진",
        "recipe_mild": "톤업: 크림드핑크3+애플리콧1",
        "recipe_severe": "말로6 + (크핑+애플)1 + 진1 (어두움 심하면 진 추가)",
    },
    "red_purple": {
        "lab": [35.0, 25.0, -8.0], "healthy": False,
        "label_ko": "Red Purple (검붉은 보라색)",
        "direction": "명도를 올리면서 동시에 검붉은 색을 따뜻한 방향으로 이동시킵니다(레드브라운과 원리 동일). 애매하면 애플리콧 비율을 줄이세요.",
        "toneup": "크림드핑크1 + 애플리콧1",
        "neutralize_direction": "웜베이지핑크 + 살몬컬러 + 진",
        "recipe_mild": "톤업: 크림드핑크1+애플리콧1",
        "recipe_severe": "로즈로얄6 + (크핑+애플)1 + 진1 (어두움 심하면 진 추가)",
    },
}


# =========================================================
# 1. 이미지 로딩 / 전처리 유틸
# =========================================================

def load_rgb_array(file_bytes: bytes, max_side: int = 500) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("이미지 파일을 읽을 수 없습니다. 손상되었거나 지원하지 않는 형식입니다.") from exc
    image.thumbnail((max_side, max_side))
    return np.array(image)


def gray_world_white_balance(img_array: np.ndarray) -> np.ndarray:
    """간이 Gray-World 화이트밸런스 보정 (조명 편차 완화)."""
    img = img_array.astype(np.float32)
    channel_avg = img.reshape(-1, 3).mean(axis=0)
    gray_avg = channel_avg.mean()
    scale = gray_avg / np.clip(channel_avg, 1e-6, None)
    return np.clip(img * scale, 0, 255).astype(np.uint8)


def _weighted_kmeans_lab(img_array: np.ndarray, n_clusters: int, seed: int = 42):
    """공통 K-means 클러스터링 헬퍼."""
    img_lab = color.rgb2lab(img_array)
    features = img_lab.copy()
    features[:, :, 0] *= 0.1
    features[:, :, 1] *= 2.0
    features[:, :, 2] *= 1.5

    pixels_features = features.reshape(-1, 3)
    pixels_original = img_lab.reshape(-1, 3)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
        labels = kmeans.fit_predict(pixels_features)
    except Exception as exc:
        raise RuntimeError("색상 군집화(K-means)에 실패했습니다. 이미지 해상도나 형식을 확인해주세요.") from exc

    centers = []
    for i in range(n_clusters):
        cluster_pixels = pixels_original[labels == i]
        centers.append(np.mean(cluster_pixels, axis=0) if len(cluster_pixels) > 0 else np.array([0.0, 0.0, 0.0]))
    centers = np.array(centers)
    return img_lab, labels, centers


def _refine_mask(mask_uint8: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels_img == largest_label).astype(bool)
    return opened.astype(bool)


# =========================================================
# 2. 핵심 분석 함수
# =========================================================

def analyze_and_mask_lip(file_bytes: bytes, use_white_balance: bool = True) -> dict:
    """시술 전 입술 접사 사진을 분석하여 메인/다크 톤 마스크와 Lab 값을 추출."""
    img_array = load_rgb_array(file_bytes)
    if use_white_balance:
        img_array = gray_world_white_balance(img_array)

    h, w, _ = img_array.shape
    img_lab, labels, actual_centers = _weighted_kmeans_lab(img_array, n_clusters=4)

    sorted_by_a_idx = np.argsort(actual_centers[:, 1])[::-1]
    lip_idx_1, lip_idx_2 = sorted_by_a_idx[0], sorted_by_a_idx[1]

    if (actual_centers[lip_idx_1][0] + actual_centers[lip_idx_1][1]) > \
       (actual_centers[lip_idx_2][0] + actual_centers[lip_idx_2][1]):
        main_idx, dark_idx = lip_idx_1, lip_idx_2
    else:
        main_idx, dark_idx = lip_idx_2, lip_idx_1

    main_lab = actual_centers[main_idx]
    dark_lab = actual_centers[dark_idx]

    labels_2d = labels.reshape(h, w)
    raw_main_mask = (labels_2d == main_idx).astype(np.uint8)
    raw_dark_mask = (labels_2d == dark_idx).astype(np.uint8)

    main_mask = _refine_mask(raw_main_mask)
    dark_mask = _refine_mask(raw_dark_mask)
    full_lip_mask = main_mask | dark_mask

    lip_area_ratio = float(np.sum(full_lip_mask)) / (h * w)
    quality_warning = lip_area_ratio < MIN_LIP_AREA_RATIO

    delta_e_main_dark = float(color.deltaE_ciede2000(main_lab, dark_lab))
    is_two_tone = bool(delta_e_main_dark > PERCEPTUAL_JND)

    skin_mask = ~full_lip_mask
    skin_tone_result = "분석 불가"
    if np.any(skin_mask):
        skin_lab = np.mean(img_lab[skin_mask], axis=0)
        sl, _sa, sb = skin_lab
        st_type = "웜톤(Warm)" if sb > 13 else "쿨톤(Cool)" if sb < 10 else "뉴트럴(Neutral)"
        st_light = "밝은 피부" if sl > 65 else "어두운 피부" if sl < 45 else "중간 피부"
        skin_tone_result = f"{st_light} / {st_type}"

    # 하얀 입술(색소 거의 없음) 안전 감지
    is_white_lip = bool(main_lab[0] > WHITE_LIP_L_THRESHOLD and main_lab[1] < WHITE_LIP_A_THRESHOLD)

    return {
        "base_img": img_array,
        "main_mask": main_mask,
        "dark_mask": dark_mask,
        "full_lip_mask": full_lip_mask,
        "main_lab": main_lab,
        "dark_lab": dark_lab,
        "is_two_tone": is_two_tone,
        "delta_e_main_dark": delta_e_main_dark,
        "skin_tone_result": skin_tone_result,
        "lip_area_ratio": lip_area_ratio,
        "quality_warning": quality_warning,
        "is_white_lip": is_white_lip,
    }


def get_single_dominant_lab(file_bytes: bytes, use_white_balance: bool = False) -> np.ndarray:
    """
    목표 색상(참고) 사진에서 가장 지배적인 색을 추출.

    ⚠️ 기본값을 use_white_balance=False로 둡니다. 이 함수가 받는 사진은 대개
    "단색 색상 스와치/참고 이미지"에 가까운데, Gray-World 보정은 "이미지 전체
    평균이 회색에 가까워야 한다"는 가정에 의존합니다. 단색에 가까운 이미지에
    이 가정을 적용하면 색 전체가 회색으로 왜곡되는 심각한 오류가 발생합니다
    (실측 테스트: 단색 [180,70,90] → 보정 후 [113,113,113]로 무채색화됨).
    반대로 analyze_and_mask_lip()이 다루는 입술 접사 사진은 피부·입술 등
    색이 섞여 있어 Gray-World 가정이 상대적으로 덜 위험하므로 그쪽은 기본 True를 유지합니다.
    """
    img_array = load_rgb_array(file_bytes, max_side=150)
    if use_white_balance:
        img_array = gray_world_white_balance(img_array)
    _img_lab, _labels, centers = _weighted_kmeans_lab(img_array, n_clusters=3)
    return np.array(max(centers, key=lambda c: c[1]))


# =========================================================
# 3. 인지 기준(ΔE00) 색상군 진단
# =========================================================

def classify_perceptual_tone(lab: np.ndarray) -> dict:
    """
    측정된 Lab 값을 레퍼런스 색상군과 ΔE00으로 비교하여 가장 가까운 카테고리를 찾음.
    """
    lab = np.asarray(lab, dtype=float)
    distances = []
    for key, ref in REFERENCE_TONES.items():
        d = float(color.deltaE_ciede2000(lab, np.array(ref["lab"], dtype=float)))
        distances.append((key, d))
    distances.sort(key=lambda x: x[1])

    best_key, best_dist = distances[0]
    second_dist = distances[1][1] if len(distances) > 1 else best_dist + CONFIDENCE_MARGIN + 1
    margin = second_dist - best_dist
    confidence = "높음" if margin >= CONFIDENCE_MARGIN else "낮음(경계선 케이스)"

    ranked = [{"key": k, "label_ko": REFERENCE_TONES[k]["label_ko"], "delta_e": d} for k, d in distances[:3]]

    ref = REFERENCE_TONES[best_key]
    return {
        "key": best_key,
        "delta_e": best_dist,
        "confidence": confidence,
        "margin": margin,
        "ranked_top3": ranked,
        **ref,
    }


def get_treatment_guide(main_lab, dark_lab, is_two_tone, is_white_lip: bool, severity: str = "mild") -> dict:
    """
    ΔE00 기반 색상군 분류 결과를 "톤업 → 중화" 2단계 시술 가이드로 변환.
    severity: "mild"(경미) 또는 "severe"(심함) — 레시피 강도 선택.
    """
    if is_white_lip:
        return {
            "mode": "white_lip",
            "warning": True,
            "message": (
                "색소가 거의 없는 '하얀 입술'로 감지되었습니다. "
                "포피·진·루루스로즈·라라·로얄레드처럼 형광기 있거나 채도 높은 컬러를 "
                "절대 먼저 사용하면 안 됩니다."
            ),
            "protocol": [
                "1차: 말로6 + 헤이즈1 로 톤다운하여 먼저 시술",
                "탈각 후 리터치 시점에 메인컬러 시술",
            ],
        }

    target_lab = dark_lab if is_two_tone else main_lab
    diagnosis = classify_perceptual_tone(target_lab)

    if diagnosis["healthy"]:
        return {
            "mode": "healthy",
            "region": "테두리(다크 톤)" if is_two_tone else "입술 전체",
            "diagnosis": diagnosis,
            "needed": False,
            "message": (
                f"가장 가까운 기준 톤은 '{diagnosis['label_ko']}'이며 "
                f"ΔE00={diagnosis['delta_e']:.2f}로 건강한 발색 범주에 해당합니다. 사전 중화 불필요."
            ),
        }

    recipe = diagnosis["recipe_severe"] if severity == "severe" else diagnosis.get("recipe_mild", diagnosis["recipe_severe"])

    return {
        "mode": "needs_correction",
        "region": "테두리(다크 톤)" if is_two_tone else "입술 전체",
        "diagnosis": diagnosis,
        "needed": True,
        "direction": diagnosis["direction"],
        "toneup": diagnosis["toneup"],
        "neutralize_direction": diagnosis["neutralize_direction"],
        "recipe": recipe,
        "severity": severity,
    }


def analyze_lip_tone_detailed(main_lab, dark_lab, is_two_tone, delta_e_main_dark):
    """참고용 서술(descriptive) 정보. 실제 판단 기준은 classify_perceptual_tone()의 ΔE00."""
    l, _a, b = main_lab
    if l < 43:
        lightness, l_reason = "어두운 톤", f"명도(L*)가 {l:.1f}로 낮아 전체적으로 어둡습니다."
    elif l > 60:
        lightness, l_reason = "밝은 톤", f"명도(L*)가 {l:.1f}로 높아 색소 발색이 유리합니다."
    else:
        lightness, l_reason = "중간 밝기 톤", f"명도(L*)가 {l:.1f}로 평균적인 밝기입니다."

    if b < 5:
        hue, h_reason = "쿨톤 (푸른기/보랏빛)", f"노란/푸른기(b*)가 {b:.1f}로 낮아 차가운 온도를 띱니다."
    elif b > 18:
        hue, h_reason = "웜톤 (오렌지/노란기)", f"노란/푸른기(b*)가 {b:.1f}로 높아 따뜻한 온도를 띱니다."
    else:
        hue, h_reason = "뉴트럴톤 (자연스러움)", f"노란/푸른기(b*)가 {b:.1f}로 중립적입니다."

    if is_two_tone:
        uni = "투톤 (테두리 착색)"
        u_reason = f"메인-테두리 ΔE00={delta_e_main_dark:.2f} (JND≈{PERCEPTUAL_JND} 초과) → 실제 착색으로 판별."
    else:
        uni = "균일한 톤"
        u_reason = f"메인-테두리 ΔE00={delta_e_main_dark:.2f} (JND≈{PERCEPTUAL_JND} 이하) → 균일한 상태로 판별."

    return lightness, l_reason, hue, h_reason, uni, u_reason


# =========================================================
# 4. 배합 최적화 (ΔE2000 기반)
# =========================================================

def find_best_mix(target_lab, pigment_db: dict):
    """목표 색상에 가장 가까운 단일/2종 배합을 ΔE2000 기준으로 탐색 (단종 색소 제외)."""
    all_pigments = iter_active_pigments(pigment_db)
    if not all_pigments:
        return None

    best_match = None
    min_delta_e = float("inf")

    for pig in all_pigments:
        delta_e = float(color.deltaE_ciede2000(target_lab, pig["lab"]))
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_match = {"p1": pig, "p2": None, "r1": 100, "r2": 0, "delta_e": delta_e}

    import itertools as _itertools
    for p1, p2 in _itertools.combinations(all_pigments, 2):
        for ratio in range(10, 100, 10):
            mixed_lab = (p1["lab"] * (ratio / 100.0)) + (p2["lab"] * ((100 - ratio) / 100.0))
            delta_e = float(color.deltaE_ciede2000(target_lab, mixed_lab))
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match = {
                    "p1": p1, "p2": p2, "r1": ratio, "r2": 100 - ratio,
                    "delta_e": delta_e, "mixed_lab": mixed_lab,
                }
    return best_match


# =========================================================
# 5. 시각화 유틸
# =========================================================

def apply_color_overlay(base_img, mask, hex_color, alpha=0.6):
    import matplotlib.colors as mcolors
    overlay = base_img.copy()
    rgb_color = mcolors.to_rgb(hex_color)
    color_uint8 = (np.array(rgb_color) * 255).astype(np.uint8)
    for c in range(3):
        overlay[mask, c] = (base_img[mask, c] * (1 - alpha) + color_uint8[c] * alpha).astype(np.uint8)
    return overlay


def generate_distribution_map(base_img, main_mask, dark_mask):
    import matplotlib.colors as mcolors
    overlay = base_img.copy()
    c_main = np.array(mcolors.to_rgb("#ff9ff3")) * 255
    c_dark = np.array(mcolors.to_rgb("#54a0ff")) * 255
    for c in range(3):
        overlay[main_mask, c] = (base_img[main_mask, c] * 0.4 + c_main[c] * 0.6).astype(np.uint8)
        overlay[dark_mask, c] = (base_img[dark_mask, c] * 0.4 + c_dark[c] * 0.6).astype(np.uint8)
    return overlay


def lab_to_hex(lab_array):
    import matplotlib.colors as mcolors
    try:
        return mcolors.to_hex(color.lab2rgb(np.array([[lab_array]]))[0][0])
    except Exception:
        return "#CCCCCC"
