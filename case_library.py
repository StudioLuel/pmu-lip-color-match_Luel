# -*- coding: utf-8 -*-
"""
case_library.py — 검증된 시술 케이스 라이브러리 관리
======================================================

[목적]
이 프로그램의 "진짜 정답"은 알고리즘의 계산 결과가 아니라,
동일 조명·환경에서 촬영되고 숙련자가 검증한 실제 시술 케이스입니다.
이 모듈은 그 케이스들을 저장/조회/필터링합니다.

[데이터 구조]
case_library/
├── index.json          ← 모든 케이스의 메타데이터 (아래 CASE 구조)
└── images/
    └── {case_id}_{slot}.jpg   (slot: before / target / mix / after)

[케이스 스키마]
{
  "case_id": "c0001",
  "created_at": "2026-10-01T14:00:00+09:00",
  "author_email": "bong03216@gmail.com",
  "environment_tag": "샵A-창가조명-2026",   # 동일 조명/환경 식별 태그 (필수)
  "color_group": "red_brown",               # REFERENCE_TONES 키 중 하나, 또는 None
  "is_two_tone": true,
  "is_white_lip": false,
  "difficulty": "중급",                      # 초급/중급/고급
  "toneup_used": "크림드핑크3+애플리콧1",
  "neutralize_used": "말로6+(크핑+애플)1+진1",
  "sessions": 2,
  "notes": "자유 메모 (실무자 코멘트, 금기사항 등)",
  "images": {
    "before": "images/c0001_before.jpg",
    "target": "images/c0001_target.jpg",
    "mix": "images/c0001_mix.jpg",
    "after": "images/c0001_after.jpg"
  },
  "verified": false   # 초급자가 입력한 미검증 케이스는 false, 숙련자 검수 후 true로 변경
}

[현재 단계에 대한 설계 노트]
사용자(초급 시술자)가 데이터를 서서히 확보해 나가는 단계이므로:
- 라이브러리가 비어 있어도 앱의 학습/연습 모드가 동작하도록, 호출부(app.py)는
  항상 "내장 지식(REFERENCE_TONES, curriculum.py)"을 1차 폴백으로 사용합니다.
- 이미지는 4장 모두 필수가 아니라 "최소 1장(before)"만 있어도 저장 가능하도록 느슨하게 설계했습니다.
  나중에 점진적으로 채워나갈 수 있습니다.
- verified 플래그를 두어, 추후 숙련자가 검수한 케이스만 "학습 모드 정답"으로 승격할 수 있는
  구조를 미리 마련해 두었습니다.
"""

import json
import os
import uuid
from datetime import datetime, timezone, timedelta

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "case_library")
IMAGES_DIR = os.path.join(LIBRARY_DIR, "images")
INDEX_PATH = os.path.join(LIBRARY_DIR, "index.json")

KST = timezone(timedelta(hours=9))


def _ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _load_index() -> list:
    _ensure_dirs()
    if not os.path.exists(INDEX_PATH):
        return []
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(cases: list):
    _ensure_dirs()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def list_cases(color_group: str = None, difficulty: str = None, verified_only: bool = False) -> list:
    """필터 조건에 맞는 케이스 목록 반환 (최신순)."""
    cases = _load_index()
    if color_group:
        cases = [c for c in cases if c.get("color_group") == color_group]
    if difficulty:
        cases = [c for c in cases if c.get("difficulty") == difficulty]
    if verified_only:
        cases = [c for c in cases if c.get("verified")]
    return sorted(cases, key=lambda c: c.get("created_at", ""), reverse=True)


def get_case(case_id: str):
    for c in _load_index():
        if c["case_id"] == case_id:
            return c
    return None


def library_stats() -> dict:
    cases = _load_index()
    by_group = {}
    for c in cases:
        g = c.get("color_group") or "미분류"
        by_group[g] = by_group.get(g, 0) + 1
    return {
        "total": len(cases),
        "verified": sum(1 for c in cases if c.get("verified")),
        "by_group": by_group,
    }


def add_case(
    author_email: str,
    environment_tag: str,
    color_group: str,
    is_two_tone: bool,
    is_white_lip: bool,
    difficulty: str,
    toneup_used: str,
    neutralize_used: str,
    sessions: int,
    notes: str,
    image_files: dict,  # {"before": bytes, "target": bytes|None, "mix": bytes|None, "after": bytes|None}
    verified: bool = False,
) -> str:
    """
    새 케이스를 라이브러리에 저장. 최소한 'before' 이미지는 필수.
    반환값: 생성된 case_id
    """
    _ensure_dirs()
    if "before" not in image_files or not image_files["before"]:
        raise ValueError("최소한 '시술 전(before)' 사진은 필수입니다.")

    case_id = f"c{uuid.uuid4().hex[:8]}"
    saved_images = {}
    for slot, file_bytes in image_files.items():
        if not file_bytes:
            continue
        ext = "jpg"
        rel_path = os.path.join("images", f"{case_id}_{slot}.{ext}")
        abs_path = os.path.join(LIBRARY_DIR, rel_path)
        with open(abs_path, "wb") as f:
            f.write(file_bytes)
        saved_images[slot] = rel_path

    case = {
        "case_id": case_id,
        "created_at": datetime.now(KST).isoformat(),
        "author_email": author_email,
        "environment_tag": environment_tag,
        "color_group": color_group,
        "is_two_tone": is_two_tone,
        "is_white_lip": is_white_lip,
        "difficulty": difficulty,
        "toneup_used": toneup_used,
        "neutralize_used": neutralize_used,
        "sessions": sessions,
        "notes": notes,
        "images": saved_images,
        "verified": verified,
    }

    cases = _load_index()
    cases.append(case)
    _save_index(cases)
    return case_id


def delete_case(case_id: str) -> bool:
    cases = _load_index()
    remaining = [c for c in cases if c["case_id"] != case_id]
    if len(remaining) == len(cases):
        return False
    # 이미지 파일도 함께 정리
    target = next((c for c in cases if c["case_id"] == case_id), None)
    if target:
        for rel_path in target.get("images", {}).values():
            abs_path = os.path.join(LIBRARY_DIR, rel_path)
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except OSError:
                    pass
    _save_index(remaining)
    return True


def image_abs_path(rel_path: str) -> str:
    return os.path.join(LIBRARY_DIR, rel_path)
