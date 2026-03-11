"""
file_renamer.py — AI 분석 제목 기반 다운로드 파일명 변경 유틸리티.

파일명 형식: [등급_평가]_소셜원본제목_또는_한국어제목.mp4
예시: [A등급_탁월]_가성비_끝판왕_햄스터_러닝머신.mp4
      [B등급_양호]_직장인_95%가_모르는_엑셀_단축키.mp4

사용 방식:
  1. 파이프라인 내부: rename_video_file(...) 호출
  2. 일괄 처리 스크립트: python -m pipeline.file_renamer [--dry-run]
"""

import os
import re
import glob
import shutil
from typing import Optional


# 파일명으로 사용할 수 없는 문자 패턴 (Windows 기준)
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|]')

# 제목 또는 파일명 어디서든 등급 추출 (A~F등급)
# 예: "A등급", "B등급_양호한_콘텐츠", "[C등급_개선..." 등
_GRADE_PATTERN = re.compile(r'(?:^\[?)?([A-F]등급)')

# 등급 → 간단 평가 매핑
_GRADE_VERDICT: dict[str, str] = {
    "A등급": "우수",
    "B등급": "양호",
    "C등급": "보통",
    "D등급": "개선필요",
    "E등급": "개선필요",
    "F등급": "개선필요",
}


def _sanitize(text: str, max_len: int = 60) -> str:
    """파일명에 사용 가능한 안전한 문자열로 변환 (띄어쓰기 유지)."""
    # 구버전 AI 제목의 "원제_Video_by_xxx" 또는 "(원본제목)" 패턴 제거
    text = re.sub(r'원제[_\s].*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\([^)]{1,50}\)\s*$', '', text).strip()
    # 특수 구분자 제거
    text = text.replace("·", "").replace("—", "").replace("–", "")
    # 빈칸 처리 (언더스코어 변환 대신 단일 공백으로 치환)
    text = re.sub(r'\s+', ' ', text.strip())
    # Windows 파일명 금지 문자 제거
    text = _INVALID_CHARS.sub('', text)
    # 혹시 남은 앞뒤 밑줄 제거
    text = text.strip('_ ')
    # 길이 제한
    return text[:max_len]


def _extract_grade_from_title(ai_title: str) -> tuple[str, str]:
    """
    AI 제목에서 등급과 간단 평가를 추출합니다.

    Returns:
        (grade, verdict) 예: ("A등급", "탁월")
        등급이 없으면 ("", "")
    """
    match = _GRADE_PATTERN.search(ai_title.strip())
    if match:
        grade = match.group(1)  # 예: "B등급"
        verdict = _GRADE_VERDICT.get(grade, "")
        return grade, verdict
    return "", ""


def build_filename(
    ai_title: str,
    source_title: str = "",
    ext: str = ".mp4",
) -> str:
    """
    '[등급_평가]_제목.mp4' 형식의 파일명을 생성합니다.

    Args:
        ai_title:     AI가 생성한 분석 제목
                      예: "A등급 · 강한 훅과 빠른 템포의 에너제틱한 숏폼 — 바이럴 잠재력 높음"
        source_title: 소셜 플랫폼 원본 제목 (없으면 ai_title 기반 한국어 설명 사용)
        ext:          파일 확장자

    Returns:
        "[A등급_탁월]_가성비_끝판왕_햄스터_러닝머신.mp4"
    """
    ai_title = ai_title.strip()

    # ── 1. 등급 & 평가 추출 → 브래킷 부분 ────────────────────────────
    grade, verdict = _extract_grade_from_title(ai_title)

    if grade and verdict:
        bracket = f"[{grade}_{verdict}]"
    elif grade:
        bracket = f"[{grade}]"
    else:
        bracket = ""

    # ── 2. 제목 본문 결정 ─────────────────────────────────────────────
    # 새로운 규칙: AI가 만들어준 한 줄 요약을 무조건 우선 사용
    # ai_title에서 등급(예: "A등급 · ") 이후 부분 추출
    m = _GRADE_PATTERN.search(ai_title)
    if m:
        body = ai_title[m.end():].strip()
        # 구버전 더미 찌꺼기 제거
        if body.startswith("_") or body.startswith("]"):
            bracket_end = body.find("]")
            if bracket_end != -1:
                body = body[bracket_end + 1:].strip().lstrip("_").strip()
            else:
                body = body.lstrip("_").strip()
        if "·" in body:
            body = body.split("·", 1)[1].strip()
        if "—" in body:
            body = body.split("—", 1)[0].strip()
    else:
        body = ai_title

    # 만약 방어 로직으로 body가 비었을 때 fallback으로 원본 제목 사용
    if not body and source_title:
        title_body = source_title.strip()
    else:
        title_body = body

    sanitized_title = _sanitize(title_body, max_len=50)

    # ── 3. 조합 ───────────────────────────────────────────────────────
    if bracket:
        # [A등급_우수] 봄동으로 비빔밥 만들기.mp4 형식 (띄어쓰기 1칸)
        filename = f"{bracket} {sanitized_title}{ext}"
    else:
        filename = f"{sanitized_title}{ext}"

    return filename


def rename_video_file(
    file_path: str,
    ai_title: str,
    source_title: str = "",
    db=None,
    video=None,
) -> Optional[str]:
    """
    단일 파일을 '[등급_평가]_제목.mp4' 형식으로 변경합니다.

    Args:
        file_path:    현재 파일 경로
        ai_title:     AI가 생성한 분석 제목
        source_title: 소셜 플랫폼 원본 제목 (선택)
        db:           SQLAlchemy 세션 (DB 업데이트 시)
        video:        Video 모델 인스턴스 (DB 업데이트 시)

    Returns:
        변경된 새 파일 경로 or None (실패 시)
    """
    if not file_path:
        return None

    # 정확한 경로가 없으면 같은 디렉터리에서 동일 stem(UUID 포함)으로 glob 탐색
    if not os.path.exists(file_path):
        dir_path_fb = os.path.dirname(os.path.abspath(file_path))
        stem_fb = os.path.splitext(os.path.basename(file_path))[0]
        matches = [
            f for f in glob.glob(os.path.join(dir_path_fb, f"{stem_fb}*"))
            if not f.endswith('.part') and os.path.isfile(f)
        ]
        if matches:
            file_path = matches[0]
            print(f"[FileRenamer] 실제 파일 발견: {os.path.basename(file_path)}")
        else:
            print(f"[FileRenamer] ⚠️  파일을 찾을 수 없음: {file_path}")
            return None

    if not ai_title or not ai_title.strip():
        print(f"[FileRenamer] ⚠️  제목이 없어 파일명 변경 불가: {file_path}")
        return None

    dir_path = os.path.dirname(file_path)
    ext = os.path.splitext(file_path)[1].lower() or ".mp4"

    new_filename = build_filename(ai_title, source_title=source_title, ext=ext)
    new_path = os.path.join(dir_path, new_filename)

    # 이미 같은 이름이면 건너뜀
    if os.path.abspath(file_path) == os.path.abspath(new_path):
        print(f"[FileRenamer] ✅ 이미 올바른 파일명: {new_filename}")
        return file_path

    # 충돌 시 숫자 suffix 추가
    if os.path.exists(new_path):
        base = os.path.splitext(new_path)[0]
        counter = 2
        while os.path.exists(new_path):
            new_path = f"{base}_{counter}{ext}"
            new_filename = os.path.basename(new_path)
            counter += 1

    try:
        shutil.move(file_path, new_path)
        print(f"[FileRenamer] ✅ {os.path.basename(file_path)}\n              → {new_filename}")

        # DB 업데이트
        if db is not None and video is not None:
            video.file_path = new_path
            db.commit()

        return new_path

    except Exception as e:
        print(f"[FileRenamer] ❌ 파일명 변경 실패 ({os.path.basename(file_path)}): {e}")
        return None


# ── 일괄 처리 (직접 실행 시) ──────────────────────────────────────────────────

def batch_rename_all(dry_run: bool = False) -> None:
    """
    DB에 있는 모든 완료된 영상의 파일명을 일괄 변경합니다.

    Args:
        dry_run: True이면 실제 변경 없이 결과만 출력합니다.
    """
    import sys

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from api.database import SessionLocal
    from api import models

    db = SessionLocal()
    try:
        videos = (
            db.query(models.Video)
            .filter(
                models.Video.status == "completed",
                models.Video.title.isnot(None),
                models.Video.file_path.isnot(None),
            )
            .all()
        )

        mode = "[DRY RUN] " if dry_run else ""
        print(f"\n{mode}총 {len(videos)}개의 완료된 영상 처리 시작\n")
        print("-" * 70)

        success, skipped, failed = 0, 0, 0

        for v in videos:
            if not v.file_path or not os.path.exists(v.file_path):
                print(f"[건너뜀] ID={v.id} 파일 없음: {v.file_path}")
                skipped += 1
                continue

            ext = os.path.splitext(v.file_path)[1].lower() or ".mp4"
            # source_title: DB에 저장된 값 우선, 없으면 빈 문자열
            src_title = getattr(v, "source_title", None) or ""
            new_filename = build_filename(v.title, source_title=src_title, ext=ext)

            dir_path = os.path.dirname(v.file_path)
            new_path = os.path.join(dir_path, new_filename)
            current_name = os.path.basename(v.file_path)

            if os.path.abspath(v.file_path) == os.path.abspath(new_path):
                print(f"[동일]   ID={v.id}  {current_name}")
                skipped += 1
                continue

            print(f"[변경]   ID={v.id}")
            print(f"  이전: {current_name}")
            print(f"  이후: {new_filename}")

            if not dry_run:
                result = rename_video_file(
                    v.file_path, v.title,
                    source_title=src_title,
                    db=db, video=v,
                )
                if result:
                    success += 1
                else:
                    failed += 1
            else:
                success += 1

            print()

        print("-" * 70)
        print(f"\n결과 요약:")
        print(f"  ✅ 변경 완료: {success}개")
        print(f"  ⏭️  건너뜀:   {skipped}개")
        print(f"  ❌ 실패:     {failed}개\n")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI 분석 결과 기반 다운로드 파일명 일괄 변경 ([등급_평가]_제목 형식)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 변경 없이 변경 내용만 미리 확인합니다.",
    )
    args = parser.parse_args()

    batch_rename_all(dry_run=args.dry_run)
