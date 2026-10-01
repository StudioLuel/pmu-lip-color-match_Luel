# -*- coding: utf-8 -*-
"""
app.py — PMU Lip Color Match: 초급 시술자 학습 플랫폼
=========================================================

[프로그램의 목적 — 중요]
이 프로그램은 "정답을 계산해주는 도구"가 아니라,
"초급 시술자가 다양한 입술 케이스를 반복적으로 진단·연습하며
색상 판단력을 체계적으로 기르는 학습 시뮬레이터"입니다.

색채과학(ΔE2000) 계산 결과는 "하나의 참고 가설"이며,
진짜 기준은 동일 조명·환경에서 촬영되고 숙련자가 검증한
케이스 라이브러리입니다. 라이브러리가 쌓일수록 이 프로그램의
학습 효과와 신뢰도가 함께 올라갑니다.

[모드 구성]
🏠 소개            — 프로그램 목적, 사용법, 현재 라이브러리 현황
📚 학습 모드        — 색상군(5종) 지식 + 저장된 실제 케이스 열람
🧠 진단 연습 모드   — 사진을 보고 먼저 판단 → AI 계산과 비교 (블라인드 연습)
🧪 샌드박스         — 자유 분석기 (사진 업로드 → 색상 진단/배합 계산), 결과를 케이스로 저장 가능
➕ 케이스 추가      — 실제 시술 케이스를 라이브러리에 축적 (동일 조명/환경 태그 필수)
"""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from skimage import color

import case_library as lib
import lip_analysis as la

st.set_page_config(page_title="PMU Lip Color Match — 학습 플랫폼", page_icon="💋", layout="centered")

st.markdown(
    """
    <style>
    .step-header { background-color: #e2e8f0; color: #111111 !important; padding: 10px; border-radius: 5px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
    .reason-text { font-size: 0.85em; color: #555555; margin-top: 2px; margin-bottom: 10px; padding-left: 10px; border-left: 2px solid #ddd; }
    .warn-box { background-color: #fff3cd; color: #664d03 !important; padding: 10px; border-radius: 5px; border: 1px solid #ffe69c; margin-bottom: 10px; }
    .danger-box { background-color: #f8d7da; color: #58151c !important; padding: 10px; border-radius: 5px; border: 1px solid #f1aeb5; margin-bottom: 10px; }
    .group-card { background-color: #f8f9fa; color: #111111 !important; padding: 12px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def color_box(hex_code, size=22):
    if not hex_code:
        hex_code = "#CCCCCC"
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'background:{hex_code};border-radius:4px;border:1px solid #999;'
        f'vertical-align:middle;margin-right:6px;"></span>'
    )


# =========================================================
# 사이드바: 모드 선택 + 공통 설정
# =========================================================

st.sidebar.title("💋 PMU 학습 플랫폼")
mode = st.sidebar.radio(
    "모드 선택",
    ["🏠 소개", "📚 학습 모드", "🧠 진단 연습 모드", "🧪 샌드박스", "➕ 케이스 추가"],
)

st.sidebar.divider()
with st.sidebar.expander("⚙️ 고급 설정"):
    use_white_balance = st.checkbox("화이트밸런스 보정 (Gray-World)", value=True)
    author_email = st.text_input("작성자 이메일(케이스 저장용)", value="bong03216@gmail.com")

stats = lib.library_stats()
st.sidebar.caption(f"📦 라이브러리: 총 {stats['total']}건 (검증 {stats['verified']}건)")


# =========================================================
# 🏠 소개
# =========================================================

if mode == "🏠 소개":
    st.title("💋 PMU Lip Color Match — 학습 플랫폼")
    st.markdown(
        """
        이 프로그램은 **반영구 입술 시술 초급자**가 다양한 입술 케이스를 반복적으로
        진단·연습하며 **색상 판단력**을 기르기 위한 학습 도구입니다.

        > ⚠️ 이 프로그램의 계산 결과(색상군 분류, 배합 추천)는 **색채과학 공식에 기반한
        > 참고 가설**입니다. 실제 정답은 숙련자가 검증한 **케이스 라이브러리**이며,
        > 최종 시술 판단은 반드시 숙련자의 육안 확인을 거쳐야 합니다.
        """
    )

    st.markdown('<div class="step-header">📈 현재 라이브러리 현황</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("전체 케이스", f"{stats['total']}건")
    c2.metric("숙련자 검증 완료", f"{stats['verified']}건")

    if stats["by_group"]:
        df = pd.DataFrame(
            [{"색상군": k, "케이스 수": v} for k, v in stats["by_group"].items()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "아직 저장된 케이스가 없습니다. '➕ 케이스 추가' 메뉴에서 실제 시술 사진과 "
            "레시피를 하나씩 쌓아가세요. 케이스가 없어도 '📚 학습 모드'의 내장 지식과 "
            "'🧠 진단 연습 모드'는 바로 사용할 수 있습니다."
        )

    st.markdown('<div class="step-header">🗺️ 사용 순서 제안</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **📚 학습 모드**에서 5가지 색상군(Yellow/Gray/Red × Brown/Purple)의 진단 논리와
           톤업·중화 레시피를 먼저 익힙니다.
        2. **🧠 진단 연습 모드**에서 사진(또는 케이스)을 보고 스스로 먼저 판단해본 뒤,
           AI 계산 결과와 비교하며 판단 근거를 검증합니다.
        3. **🧪 샌드박스**에서 자유롭게 사진을 올려 분석해보고, 신뢰할 만한 결과는
           **➕ 케이스 추가**를 통해 라이브러리에 저장합니다.
        4. 라이브러리가 쌓일수록 학습 모드·연습 모드의 콘텐츠가 실제 데이터 기반으로 풍부해집니다.
        """
    )


# =========================================================
# 📚 학습 모드
# =========================================================

elif mode == "📚 학습 모드":
    st.title("📚 학습 모드")
    st.caption("5가지 색상군의 진단 논리와 톤업·중화 레시피를 먼저 익히세요. (반영구 시술 교재 기반)")

    tab1, tab2 = st.tabs(["🎨 색상군 지식", "🗂️ 저장된 실제 케이스"])

    with tab1:
        problem_groups = {k: v for k, v in la.REFERENCE_TONES.items() if not v["healthy"]}
        for key, ref in problem_groups.items():
            with st.expander(f"🔸 {ref['label_ko']}", expanded=False):
                hex_preview = la.lab_to_hex(ref["lab"])
                st.markdown(
                    f"{color_box(hex_preview, 30)} **대표 색조 미리보기** (Lab 기준점 근사치)",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**시술 방향:** {ref['direction']}")
                st.markdown(f"**1차 톤업 색상:** {ref['toneup']}")
                st.markdown(f"**중화 방향:** {ref['neutralize_direction']}")
                st.markdown("**레시피 (경미한 경우):**")
                st.code(ref.get("recipe_mild", "-"), language=None)
                st.markdown("**레시피 (심한 경우):**")
                st.code(ref.get("recipe_severe", "-"), language=None)

        with st.expander("⚠️ 특수 케이스: 하얀 입술 (색소 거의 없음)"):
            st.markdown(
                """
                - **절대 금지**: 포피·진·루루스로즈·라라·로얄레드처럼 형광기 있거나 채도 높은 컬러를 먼저 사용
                - **1차**: 말로6 + 헤이즈1 로 톤다운하여 먼저 시술
                - 탈각 후 **리터치 시점**에 메인컬러 시술
                """
            )
        with st.expander("⚠️ 특수 케이스: 잔흔 입술 / 군집성 포다이스반"):
            st.markdown(
                """
                **잔흔(얼룩) 입술**
                - 얼룩이 심한 경우: 얼룩 중 가장 진한 컬러로 전체를 균일하게 맞추고, 처음 시술하듯 진행
                - 안쪽만 빠진 경우: 잔흔과 동일하거나 살짝 더 진한 컬러로 빈 부분 위주 시술
                - **금기**: 전체가 진한 진달래색인데 누디 요청 / 진한 핑크인데 코랄로 변경 요청 → 상담 필요

                **군집성 포다이스반 동반 다크립**
                1. 반점 먼저 시술 (말로6 + 헤이즈1)
                2. 다크립 톤업 및 중화
                3. 메인컬러 작업 (단, 어두운 부분에는 메인컬러 작업 금지)
                - 첫날 원하는 색을 바로 맞춰줄 수 없음을 반드시 사전 상담해야 함
                """
            )

        st.markdown('<div class="step-header">🧪 색소 데이터베이스 (실무 명칭)</div>', unsafe_allow_html=True)
        rows = []
        for brand, pigs in la.PIGMENT_DB.items():
            for name, meta in pigs.items():
                rows.append({
                    "브랜드": brand, "색상명": name,
                    "상태": "✅ 사용가능" if meta["status"] == "active" else "⛔ 단종",
                    "대체색상": meta.get("replacement", "-"),
                    "비고": meta.get("note", ""),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("※ Lab 값은 색상 설명 기반 추정치입니다. 실측 데이터로 교체 시 정확도가 향상됩니다.")

    with tab2:
        saved_cases = lib.list_cases()
        if not saved_cases:
            st.info("아직 저장된 실제 케이스가 없습니다. '➕ 케이스 추가'에서 하나씩 쌓아보세요.")
        else:
            group_filter = st.selectbox(
                "색상군 필터",
                ["전체"] + list({c.get("color_group") or "미분류" for c in saved_cases}),
            )
            for c in saved_cases:
                if group_filter != "전체" and (c.get("color_group") or "미분류") != group_filter:
                    continue
                label = la.REFERENCE_TONES.get(c.get("color_group"), {}).get("label_ko", c.get("color_group") or "미분류")
                verified_badge = "✅ 검증됨" if c.get("verified") else "🕗 미검증"
                with st.expander(f"{label} · {c['environment_tag']} · {verified_badge}"):
                    cols = st.columns(4)
                    slot_labels = {"before": "시술 전", "target": "목표색상", "mix": "배합색상", "after": "시술 후"}
                    for i, (slot, slabel) in enumerate(slot_labels.items()):
                        img_rel = c.get("images", {}).get(slot)
                        if img_rel:
                            cols[i].image(lib.image_abs_path(img_rel), caption=slabel, use_container_width=True)
                        else:
                            cols[i].caption(f"({slabel} 없음)")
                    st.markdown(f"**톤업:** {c.get('toneup_used', '-')}")
                    st.markdown(f"**중화:** {c.get('neutralize_used', '-')}")
                    st.markdown(f"**시술 횟수:** {c.get('sessions', '-')}회")
                    if c.get("notes"):
                        st.markdown(f"<div class='reason-text'>{c['notes']}</div>", unsafe_allow_html=True)


# =========================================================
# 🧠 진단 연습 모드 (블라인드 진단)
# =========================================================

elif mode == "🧠 진단 연습 모드":
    st.title("🧠 진단 연습 모드")
    st.caption("사진을 보고 먼저 스스로 진단해보세요. 그 다음 AI 계산 결과와 비교합니다.")

    saved_cases = lib.list_cases()

    source = st.radio(
        "연습 소스 선택",
        ["저장된 실제 케이스로 연습", "새 사진 업로드로 연습"],
        horizontal=True,
    )

    practice_bytes = None
    practice_case = None

    if source == "저장된 실제 케이스로 연습":
        if not saved_cases:
            st.warning("아직 저장된 케이스가 없습니다. '새 사진 업로드로 연습'을 이용하거나, 케이스를 먼저 추가해주세요.")
        else:
            case_options = {f"{c['case_id']} · {c['environment_tag']}": c for c in saved_cases}
            picked = st.selectbox("연습할 케이스 선택", list(case_options.keys()))
            practice_case = case_options[picked]
            before_rel = practice_case.get("images", {}).get("before")
            if before_rel:
                with open(lib.image_abs_path(before_rel), "rb") as f:
                    practice_bytes = f.read()
    else:
        uploaded = st.file_uploader("연습용 입술 사진 업로드", type=["jpg", "jpeg", "png"])
        if uploaded:
            practice_bytes = uploaded.getvalue()

    if practice_bytes:
        st.image(practice_bytes, caption="이 입술을 진단해보세요", use_container_width=True)

        st.markdown("### 1️⃣ 당신의 진단")
        with st.form("blind_diagnosis_form"):
            user_group = st.selectbox(
                "이 입술은 어떤 색상군에 가장 가깝다고 생각하나요?",
                ["healthy_warm", "healthy_cool"] + [k for k, v in la.REFERENCE_TONES.items() if not v["healthy"]],
                format_func=lambda k: la.REFERENCE_TONES[k]["label_ko"],
            )
            user_toneup = st.text_input("톤업 색상은 무엇이 적합할까요? (선택)")
            user_neutralize = st.text_input("중화 색상/방향은 무엇이 적합할까요? (선택)")
            submitted = st.form_submit_button("제출하고 AI 계산과 비교하기")

        if submitted:
            try:
                with st.spinner("AI가 ΔE2000 기준으로 색상을 분석 중입니다..."):
                    result = la.analyze_and_mask_lip(practice_bytes, use_white_balance=use_white_balance)
            except (ValueError, RuntimeError) as exc:
                st.error(f"분석 오류: {exc}")
                st.stop()

            ai_guide = la.get_treatment_guide(
                result["main_lab"], result["dark_lab"], result["is_two_tone"], result["is_white_lip"],
            )
            ai_group_key = ai_guide.get("diagnosis", {}).get("key") if ai_guide["mode"] != "white_lip" else "white_lip"

            st.markdown("### 2️⃣ 비교 결과")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🙋 당신의 진단**")
                st.info(la.REFERENCE_TONES.get(user_group, {}).get("label_ko", user_group))
                if user_toneup:
                    st.markdown(f"톤업: {user_toneup}")
                if user_neutralize:
                    st.markdown(f"중화: {user_neutralize}")
            with c2:
                st.markdown("**🤖 AI 계산 (ΔE2000 기준)**")
                if ai_guide["mode"] == "white_lip":
                    st.warning(ai_guide["message"])
                elif ai_guide["mode"] == "healthy":
                    st.success(ai_guide["message"])
                else:
                    diag = ai_guide["diagnosis"]
                    st.info(f"{diag['label_ko']} (ΔE00={diag['delta_e']:.2f}, 신뢰도: {diag['confidence']})")
                    st.markdown(f"톤업: {ai_guide['toneup']}")
                    st.markdown(f"중화: {ai_guide['neutralize_direction']}")
                    st.code(ai_guide["recipe"], language=None)

            if user_group == ai_group_key:
                st.success("✅ 당신의 진단이 AI 계산 결과와 일치합니다! 좋은 판단입니다.")
            else:
                st.markdown(
                    '<div class="warn-box">⚠️ 당신의 진단과 AI 계산이 다릅니다. '
                    "아래 근거(ΔE00, Top-3 후보)를 보며 어떤 포인트를 놓쳤는지 복기해보세요.</div>",
                    unsafe_allow_html=True,
                )
                if ai_guide["mode"] not in ("white_lip",):
                    diag = ai_guide.get("diagnosis") or la.classify_perceptual_tone(result["main_lab"])
                    st.markdown("**AI가 고려한 Top-3 후보 (ΔE00 가까운 순):**")
                    for cand in diag["ranked_top3"]:
                        st.markdown(f"- {cand['label_ko']} (ΔE00={cand['delta_e']:.2f})")

            if practice_case:
                st.divider()
                st.markdown("### 3️⃣ 실제 시술 기록 (검증된 정답)")
                st.markdown(f"**실제 사용 톤업:** {practice_case.get('toneup_used', '-')}")
                st.markdown(f"**실제 사용 중화:** {practice_case.get('neutralize_used', '-')}")
                if practice_case.get("notes"):
                    st.markdown(f"<div class='reason-text'>{practice_case['notes']}</div>", unsafe_allow_html=True)
                after_rel = practice_case.get("images", {}).get("after")
                if after_rel:
                    st.image(lib.image_abs_path(after_rel), caption="실제 시술 후", use_container_width=True)


# =========================================================
# 🧪 샌드박스 (자유 분석기)
# =========================================================

elif mode == "🧪 샌드박스":
    st.title("🧪 샌드박스 — 자유 분석기")
    st.caption("가설 검증용 자유 실험 공간입니다. 결과가 신뢰할 만하면 라이브러리에 저장해 데이터를 쌓아보세요.")

    col1, col2 = st.columns(2)
    with col1:
        current_file = st.file_uploader("시술 전 입술 확대 사진", type=["jpg", "jpeg", "png"], key="sb_current")
    with col2:
        target_file = st.file_uploader("원하는 컬러 사진 (선택)", type=["jpg", "jpeg", "png"], key="sb_target")

    if current_file:
        current_bytes = current_file.getvalue()
        try:
            with st.spinner("분석 중..."):
                result = la.analyze_and_mask_lip(current_bytes, use_white_balance=use_white_balance)
        except (ValueError, RuntimeError) as exc:
            st.error(f"분석 오류: {exc}")
            st.stop()

        base_img = result["base_img"]
        main_mask, dark_mask, full_lip_mask = result["main_mask"], result["dark_mask"], result["full_lip_mask"]
        curr_main_lab, curr_dark_lab = result["main_lab"], result["dark_lab"]
        is_two_tone = result["is_two_tone"]

        if result["quality_warning"]:
            st.markdown(
                f'<div class="warn-box">⚠️ 입술 영역이 전체의 {result["lip_area_ratio"]*100:.1f}%로 작게 '
                f"추출되었습니다. 입술이 프레임을 채우는 사진으로 다시 시도해보세요.</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="step-header">🔍 색상 분포 & 피부톤</div>', unsafe_allow_html=True)
        st.info(f"피부톤 힌트: {result['skin_tone_result']}")
        c1, c2 = st.columns(2)
        c1.image(base_img, caption="원본", use_container_width=True)
        c2.image(la.generate_distribution_map(base_img, main_mask, dark_mask), caption="분포도", use_container_width=True)

        lightness, l_reason, hue, h_reason, uni, u_reason = la.analyze_lip_tone_detailed(
            curr_main_lab, curr_dark_lab, is_two_tone, result["delta_e_main_dark"]
        )
        st.markdown(f"**명도:** {lightness} · **색온도:** {hue} · **균일도:** {uni}")
        st.markdown(f"<div class='reason-text'>{l_reason}<br>{h_reason}<br>{u_reason}</div>", unsafe_allow_html=True)

        st.markdown('<div class="step-header">🛠️ 톤업 → 중화 가이드</div>', unsafe_allow_html=True)
        severity = st.radio("증상 정도", ["mild", "severe"], format_func=lambda x: "경미" if x == "mild" else "심함", horizontal=True)
        guide = la.get_treatment_guide(curr_main_lab, curr_dark_lab, is_two_tone, result["is_white_lip"], severity=severity)

        if guide["mode"] == "white_lip":
            st.markdown(f'<div class="danger-box">🚫 {guide["message"]}</div>', unsafe_allow_html=True)
            for step in guide["protocol"]:
                st.markdown(f"- {step}")
        elif guide["mode"] == "healthy":
            st.success(guide["message"])
        else:
            diag = guide["diagnosis"]
            st.warning(f"**진단:** {diag['label_ko']} (ΔE00={diag['delta_e']:.2f}, 신뢰도: {diag['confidence']})")
            st.markdown(f"**방향:** {guide['direction']}")
            st.markdown(f"**톤업:** {guide['toneup']}")
            st.markdown(f"**중화 방향:** {guide['neutralize_direction']}")
            st.code(guide["recipe"], language=None)
            if diag["confidence"] != "높음":
                st.caption("⚠️ 경계선 케이스입니다 — Top-3 후보를 함께 검토하세요.")
                for cand in diag["ranked_top3"]:
                    st.caption(f"- {cand['label_ko']} (ΔE00={cand['delta_e']:.2f})")

        mix_hex = None
        if target_file:
            target_bytes = target_file.getvalue()
            # 목표 색상 사진은 단색 스와치에 가까우므로 화이트밸런스 보정을 적용하지 않습니다
            # (lip_analysis.get_single_dominant_lab 문서 참조).
            targ_lab = la.get_single_dominant_lab(target_bytes)
            best_mix = la.find_best_mix(targ_lab, la.PIGMENT_DB)
            if best_mix:
                st.markdown('<div class="step-header">🎨 본 컬러 배합 (ΔE2000 최소화)</div>', unsafe_allow_html=True)
                p1, p2 = best_mix["p1"], best_mix["p2"]
                if p2 is None:
                    st.success(f"단일 색상: {p1['brand']} - {p1['name']} (100%)")
                    mix_hex = la.lab_to_hex(p1["lab"])
                else:
                    st.markdown(f"**배합:** {p1['name']} {best_mix['r1']}% + {p2['name']} {best_mix['r2']}% "
                                f"(ΔE00={best_mix['delta_e']:.2f})")
                    mix_hex = mcolors.to_hex(color.lab2rgb(np.array([[best_mix["mixed_lab"]]]))[0][0])
                st.image(
                    la.apply_color_overlay(base_img, full_lip_mask, mix_hex, alpha=0.45),
                    caption="예상 결과 시뮬레이션", use_container_width=True,
                )

        st.divider()
        st.markdown('<div class="step-header">💾 이 결과를 라이브러리에 저장</div>', unsafe_allow_html=True)
        st.caption(
            "신뢰할 만한 결과라면 케이스로 저장해 데이터를 쌓아보세요. "
            "저장 시 환경 태그(촬영 장소/조명)를 꼭 입력해 데이터 신뢰성을 높여주세요."
        )
        with st.form("save_from_sandbox"):
            env_tag = st.text_input("환경 태그 (예: 샵A-창가조명-2026)")
            difficulty = st.selectbox("난이도", ["초급", "중급", "고급"])
            default_group = guide.get("diagnosis", {}).get("key") if guide["mode"] == "needs_correction" else (
                "white_lip" if guide["mode"] == "white_lip" else "healthy_warm"
            )
            notes = st.text_area("메모 (선택)")
            save_clicked = st.form_submit_button("라이브러리에 저장")

        if save_clicked:
            if not env_tag:
                st.error("환경 태그는 필수입니다. 동일 조명/환경 추적을 위해 꼭 입력해주세요.")
            else:
                case_id = lib.add_case(
                    author_email=author_email,
                    environment_tag=env_tag,
                    color_group=default_group,
                    is_two_tone=is_two_tone,
                    is_white_lip=result["is_white_lip"],
                    difficulty=difficulty,
                    toneup_used=guide.get("toneup", "-"),
                    neutralize_used=guide.get("recipe", guide.get("message", "-")),
                    sessions=1,
                    notes=notes,
                    image_files={
                        "before": current_bytes,
                        "target": target_file.getvalue() if target_file else None,
                        "mix": None,
                        "after": None,
                    },
                    verified=False,
                )
                st.success(f"케이스 {case_id}로 저장되었습니다. '📚 학습 모드'와 '🧠 진단 연습 모드'에서 확인할 수 있습니다.")


# =========================================================
# ➕ 케이스 추가 (전용 폼)
# =========================================================

elif mode == "➕ 케이스 추가":
    st.title("➕ 케이스 추가")
    st.caption(
        "실제 시술 케이스를 라이브러리에 축적합니다. 동일 조명·환경에서 촬영한 사진을 사용하면 "
        "데이터 신뢰성이 올라갑니다. 지금은 사진이 부족해도 괜찮습니다 — '시술 전' 사진 1장만 있어도 저장 가능합니다."
    )

    with st.form("add_case_form"):
        st.markdown("**1. 사진 업로드 (시술 전은 필수, 나머지는 있는 만큼만)**")
        c1, c2, c3, c4 = st.columns(4)
        before_f = c1.file_uploader("시술 전", type=["jpg", "jpeg", "png"], key="add_before")
        target_f = c2.file_uploader("목표 색상", type=["jpg", "jpeg", "png"], key="add_target")
        mix_f = c3.file_uploader("배합 색상", type=["jpg", "jpeg", "png"], key="add_mix")
        after_f = c4.file_uploader("시술 후", type=["jpg", "jpeg", "png"], key="add_after")

        st.markdown("**2. 촬영 환경 (데이터 신뢰성 핵심 — 필수)**")
        env_tag = st.text_input(
            "환경 태그", placeholder="예: 샵A-창가조명-2026, 샵A-링라이트-2026",
            help="같은 조명/장소에서 촬영한 사진들은 동일한 태그를 사용하세요. 이는 추후 Lab 값 보정의 기준이 됩니다.",
        )

        st.markdown("**3. 분류 및 레시피**")
        color_group = st.selectbox(
            "색상군",
            list(la.REFERENCE_TONES.keys()) + ["white_lip", "jan_heun", "etc"],
            format_func=lambda k: la.REFERENCE_TONES[k]["label_ko"] if k in la.REFERENCE_TONES else
            {"white_lip": "하얀 입술", "jan_heun": "잔흔 입술", "etc": "기타"}[k],
        )
        is_two_tone = st.checkbox("투톤(테두리 착색) 케이스인가요?")
        is_white_lip = st.checkbox("하얀 입술(색소 거의 없음) 케이스인가요?")
        difficulty = st.selectbox("난이도", ["초급", "중급", "고급"])
        toneup_used = st.text_input("실제 사용한 톤업 색상/레시피")
        neutralize_used = st.text_input("실제 사용한 중화 색상/레시피")
        sessions = st.number_input("총 시술 횟수", min_value=1, max_value=5, value=1)
        notes = st.text_area("메모 (진단 근거, 금기사항, 특이사항 등)")
        verified = st.checkbox("숙련자가 검수한 케이스입니다 (검증 완료로 표시)")

        submitted = st.form_submit_button("케이스 저장")

    if submitted:
        if not env_tag:
            st.error("환경 태그는 필수입니다.")
        elif not before_f:
            st.error("'시술 전' 사진은 필수입니다.")
        else:
            case_id = lib.add_case(
                author_email=author_email,
                environment_tag=env_tag,
                color_group=color_group,
                is_two_tone=is_two_tone,
                is_white_lip=is_white_lip,
                difficulty=difficulty,
                toneup_used=toneup_used,
                neutralize_used=neutralize_used,
                sessions=int(sessions),
                notes=notes,
                image_files={
                    "before": before_f.getvalue() if before_f else None,
                    "target": target_f.getvalue() if target_f else None,
                    "mix": mix_f.getvalue() if mix_f else None,
                    "after": after_f.getvalue() if after_f else None,
                },
                verified=verified,
            )
            st.success(f"✅ 케이스 {case_id}가 저장되었습니다!")
            st.balloons()
