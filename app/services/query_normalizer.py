from __future__ import annotations

import re
import unicodedata

# 목적: 캐시 hit를 조금 올리되 의미 훼손/충돌(오염) 최소화
# 전략:
# 1) NFKC + lower + 공백 정리 (필수)
# 2) 따옴표류만 제거 (입력 변형이 많고 의미 영향이 적음)
# 3) URL/FQCN/경로/버전 의미를 깨는 ., :, /, -, _ 등은 보존

_MULTI_SPACE_RE = re.compile(r"\s+")
_QUOTES_RE = re.compile(r"[\"'`“”‘’]+")

_DOMAIN_CANON_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bspring\s*boot\b", re.IGNORECASE), "spring-boot"),
    (re.compile(r"\bspringboot\b", re.IGNORECASE), "spring-boot"),
    (re.compile(r"\bspring\s*security\b", re.IGNORECASE), "spring-security"),
    (re.compile(r"\bspringsecurity\b", re.IGNORECASE), "spring-security"),
    (re.compile(r"\bspring\s*data\b", re.IGNORECASE), "spring-data"),
    (re.compile(r"\bspring\s*framework\b", re.IGNORECASE), "spring-framework"),
)


def _normalize_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _apply_domain_canon(text: str) -> str:
    out = text
    for pattern, replacement in _DOMAIN_CANON_RULES:
        out = pattern.sub(replacement, out)
    return out


def normalize_query(text: str) -> str:
    """
    사용자 query 정규화 (운영 캐시 hit 목적, 보수적)

    - 유니코드 정규화: NFKC
    - 소문자화
    - 따옴표/백틱류 제거 (의미 훼손 적고 입력 변형 많음)
    - 도메인 명칭 최소 통일 (선택)
    - 공백 정리
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    out = unicodedata.normalize("NFKC", raw)
    out = out.lower()
    out = _apply_domain_canon(out)
    out = _QUOTES_RE.sub(" ", out)
    out = _normalize_whitespace(out)
    return out
