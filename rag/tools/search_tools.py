"""
Tools for Vertex AI Search (Discovery Engine) using ADK FunctionTool pattern.

- vertex_search(): returns search results + snippets + extractive answers/segments + summary (when available)
- list_search_engines(): lists available Vertex AI Search apps (engines)
"""

from typing import Dict, Any, Optional, List, Tuple, Callable

import re
import calendar

from dataclasses import dataclass

from google.adk.tools import FunctionTool
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions

from rag.config import (
    PROJECT_ID,
    APP_LOCATION,

    # Search defaults
    SEARCH_DEFAULT_SERVING_CONFIG_ID,
    SEARCH_DEFAULT_PAGE_SIZE,
    SEARCH_DEFAULT_SUMMARY_RESULT_COUNT,
    SEARCH_DEFAULT_INCLUDE_CITATIONS,
    SEARCH_DEFAULT_USE_SEMANTIC_CHUNKS,
    SEARCH_DEFAULT_SUMMARY_MODEL_VERSION,
    SEARCH_DEFAULT_SUMMARY_PREAMBLE,
    SEARCH_DEFAULT_MAX_SNIPPET_COUNT,
    SEARCH_DEFAULT_MAX_EXTRACTIVE_ANSWER_COUNT,
    SEARCH_DEFAULT_MAX_EXTRACTIVE_SEGMENT_COUNT,
    SEARCH_DEFAULT_QUERY_EXPANSION,
    SEARCH_DEFAULT_SPELL_CORRECTION,
)

# -------------------------
# Config helpers
# -------------------------

def _get_search_location() -> str:
    return APP_LOCATION or "global"


def _build_serving_config(engine_id: str, serving_config_id: str) -> str:
    loc = _get_search_location()
    return (
        f"projects/{PROJECT_ID}/locations/{loc}/collections/default_collection/"
        f"engines/{engine_id}/servingConfigs/{serving_config_id}"
    )


def _discoveryengine_endpoint(location: str) -> Optional[str]:
    if not location or location == "global":
        return None
    return f"{location}-discoveryengine.googleapis.com"


def _query_expansion_condition(value: str):
    v = (value or "").upper()
    if v == "DISABLED":
        return discoveryengine.SearchRequest.QueryExpansionSpec.Condition.DISABLED
    return discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO


def _spell_correction_mode(value: str):
    v = (value or "").upper()
    if v == "DISABLED":
        return discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.DISABLED
    return discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO


# -------------------------
# Parsing helpers
# -------------------------

def _get_in(d: Any, path: List[str]) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_non_empty(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _doc_best_effort(doc_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (title, uri, link) from doc_dict across common schemas."""
    if not isinstance(doc_dict, dict):
        return None, None, None

    title = _first_non_empty(
        _get_in(doc_dict, ["structData", "title"]),
        _get_in(doc_dict, ["derivedStructData", "title"]),
        doc_dict.get("title"),
        doc_dict.get("displayName"),
        doc_dict.get("display_name"),
        doc_dict.get("name"),
        doc_dict.get("id"),
    )

    uri = _first_non_empty(
        _get_in(doc_dict, ["content", "uri"]),
        _get_in(doc_dict, ["derivedStructData", "link"]),
        _get_in(doc_dict, ["structData", "link"]),
        doc_dict.get("uri"),
        doc_dict.get("link"),
    )

    link = None
    if isinstance(uri, str) and uri.startswith(("http://", "https://")):
        link = uri
    else:
        link = _first_non_empty(
            doc_dict.get("link"),
            _get_in(doc_dict, ["derivedStructData", "link"]),
            _get_in(doc_dict, ["structData", "link"]),
        )

    return title, uri, link


def _extract_snippets_and_extractive(raw_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort extraction from result.to_dict()"""
    if not isinstance(raw_result, dict):
        return {"snippets": [], "extractive_answers": [], "extractive_segments": []}

    snippets: List[str] = []
    extractive_answers: List[Dict[str, Any]] = []
    extractive_segments: List[Dict[str, Any]] = []

    for cand in (
        _get_in(raw_result, ["document", "snippets"]),
        raw_result.get("snippets"),
    ):
        if isinstance(cand, list):
            for s in cand:
                if isinstance(s, dict):
                    txt = _first_non_empty(s.get("snippet"), s.get("text"))
                    if isinstance(txt, str):
                        snippets.append(txt)
                elif isinstance(s, str):
                    snippets.append(s)

    for cand in (
        _get_in(raw_result, ["document", "extractiveAnswers"]),
        raw_result.get("extractiveAnswers"),
        _get_in(raw_result, ["document", "extractive_answers"]),
        raw_result.get("extractive_answers"),
    ):
        if isinstance(cand, list):
            for a in cand:
                if isinstance(a, dict):
                    extractive_answers.append(a)

    for cand in (
        _get_in(raw_result, ["document", "extractiveSegments"]),
        raw_result.get("extractiveSegments"),
        _get_in(raw_result, ["document", "extractive_segments"]),
        raw_result.get("extractive_segments"),
    ):
        if isinstance(cand, list):
            for seg in cand:
                if isinstance(seg, dict):
                    extractive_segments.append(seg)

    # dedup snippets
    dedup_snips = []
    seen = set()
    for s in snippets:
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            dedup_snips.append(s)

    return {
        "snippets": dedup_snips[:10],
        "extractive_answers": extractive_answers[:10],
        "extractive_segments": extractive_segments[:10],
    }


def _extract_summary_text(resp: discoveryengine.SearchResponse) -> Optional[str]:
    """
    SearchResponse.summary에서 바로 요약 텍스트를 뽑는다.
    (이게 가장 안정적으로 잘 들어오는 케이스가 많음)
    """
    try:
        summary = getattr(resp, "summary", None)
        if summary is None:
            return None

        # proto object fields
        for key in ("summary_text", "summaryText", "text", "summary"):
            if hasattr(summary, key):
                val = getattr(summary, key)
                if isinstance(val, str) and val.strip():
                    return val

        # dict fallback
        if hasattr(summary, "to_dict"):
            d = summary.to_dict()
            for key in ("summaryText", "summary_text", "text", "summary"):
                val = d.get(key)
                if isinstance(val, str) and val.strip():
                    return val

    except Exception:
        pass
    return None


def _extract_summary_citations(resp: discoveryengine.SearchResponse) -> Any:
    """
    요약에 citations가 붙는 경우에만 내려옴(항상 있는 건 아님).
    """
    try:
        summary = getattr(resp, "summary", None)
        if summary is None:
            return None

        for key in ("summary_citations", "summaryCitations", "citations", "references"):
            if hasattr(summary, key):
                obj = getattr(summary, key)
                return obj.to_dict() if hasattr(obj, "to_dict") else obj

        if hasattr(summary, "to_dict"):
            d = summary.to_dict()
            for key in ("summaryCitations", "summary_citations", "citations", "references"):
                if key in d:
                    return d.get(key)

    except Exception:
        pass
    return None


def _build_citation_items_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    summary citations가 없을 때도 "Sources 목록"을 만들 수 있게 results의 source로 citations 생성.
    """
    citations: List[Dict[str, Any]] = []
    seen = set()
    idx = 0

    for r in results:
        src = r.get("source") or {}
        title = src.get("title")
        uri = src.get("uri")
        link = src.get("link")

        key = (uri or link or title or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)

        idx += 1
        citations.append(
            {
                "index": idx,
                "label": f"[{idx}]",
                "title": title,
                "uri": uri,
                "link": link,
            }
        )

    return citations


# -------------------------
# Tool 1) Search (+ summary/snippets/extractive)
# -------------------------

def vertex_search(
    engine_id: str,
    query_text: str,

    page_size: Optional[int] = None,
    filter_expr: Optional[str] = None,

    # ✅ paging / budgeting controls (NEW)
    page_token: Optional[str] = None,
    max_pages: int = 3,
    max_total_chars: int = 80_000,
    max_total_results: int = 200,

    # ✅ optionally force search result mode (NEW): "DOCUMENTS" / "CHUNKS"
    search_result_mode: Optional[str] = None,

    summary_result_count: Optional[int] = None,
    include_citations: Optional[bool] = None,
    use_semantic_chunks: Optional[bool] = None,
    summary_model_version: Optional[str] = None,
    preamble: Optional[str] = None,

    max_snippet_count: Optional[int] = None,
    max_extractive_answer_count: Optional[int] = None,
    max_extractive_segment_count: Optional[int] = None,

    query_expansion: Optional[str] = None,   # "AUTO"/"DISABLED"
    spell_correction: Optional[str] = None,  # "AUTO"/"DISABLED"

    serving_config_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Vertex AI Search: 검색 + 요약 + 스니펫 + 추출 답/구간.

    개선 포인트
    - next_page_token을 사용해 여러 페이지를 수집 (page_size가 작아도 coverage 확보)
    - 근거 수집 예산(max_total_chars/max_total_results/max_pages) 초과 시:
      is_truncated=True 로 표시하고, 사용자가 검색 범위를 줄이도록 안내
    - 필요 시 search_result_mode를 "CHUNKS"로 설정 가능

    summary_text는 resp.summary에서 뽑는다(기존 유지).
    citations는 (1) summary citations가 있으면 그것도 내려주고,
                (2) 최소한 results에서 sources 목록을 만들어 항상 제공한다.
    """
    try:
        # ---- defaults ----
        if serving_config_id is None:
            serving_config_id = SEARCH_DEFAULT_SERVING_CONFIG_ID

        if page_size is None:
            page_size = SEARCH_DEFAULT_PAGE_SIZE

        # Discovery Engine 관행상 page_size는 100 이하로 안전하게 제한
        try:
            page_size_int = int(page_size)
        except Exception:
            page_size_int = SEARCH_DEFAULT_PAGE_SIZE
        page_size_int = max(1, min(page_size_int, 100))

        if summary_result_count is None:
            summary_result_count = SEARCH_DEFAULT_SUMMARY_RESULT_COUNT
        if include_citations is None:
            include_citations = SEARCH_DEFAULT_INCLUDE_CITATIONS
        if use_semantic_chunks is None:
            use_semantic_chunks = SEARCH_DEFAULT_USE_SEMANTIC_CHUNKS
        if summary_model_version is None:
            summary_model_version = SEARCH_DEFAULT_SUMMARY_MODEL_VERSION
        if preamble is None:
            preamble = SEARCH_DEFAULT_SUMMARY_PREAMBLE

        if max_snippet_count is None:
            max_snippet_count = SEARCH_DEFAULT_MAX_SNIPPET_COUNT
        if max_extractive_answer_count is None:
            max_extractive_answer_count = SEARCH_DEFAULT_MAX_EXTRACTIVE_ANSWER_COUNT
        if max_extractive_segment_count is None:
            max_extractive_segment_count = SEARCH_DEFAULT_MAX_EXTRACTIVE_SEGMENT_COUNT

        if query_expansion is None:
            query_expansion = SEARCH_DEFAULT_QUERY_EXPANSION
        if spell_correction is None:
            spell_correction = SEARCH_DEFAULT_SPELL_CORRECTION

        # ---- client / request base ----
        loc = _get_search_location()
        serving_config = _build_serving_config(engine_id, serving_config_id)

        client = discoveryengine.SearchServiceClient(
            client_options=ClientOptions(api_endpoint=_discoveryengine_endpoint(loc))
        )

        # content spec (same as before)
        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
                max_snippet_count=max_snippet_count
            ),
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=max_extractive_answer_count,
                max_extractive_segment_count=max_extractive_segment_count,
                # allowlisted DS에만 의미 있을 수 있으므로, 필요하면 config로 토글하는 걸 권장
                return_extractive_segment_score=True,
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=summary_result_count,
                include_citations=include_citations,
                ignore_adversarial_query=True,
                use_semantic_chunks=use_semantic_chunks,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                    version=summary_model_version
                ),
                model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                    preamble=preamble
                ),
            ),
        )

        # ---- paging loop ----
        results: List[Dict[str, Any]] = []
        total_chars = 0
        pages_fetched = 0

        is_truncated = False
        truncation_reason = None

        next_token = (page_token or "").strip()

        # summary는 첫 페이지 응답에서만 뽑는게 일반적으로 충분(비용/일관성)
        summary_text = None
        summary_citations = None

        while True:
            pages_fetched += 1

            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query_text,
                page_size=page_size_int,
                content_search_spec=content_search_spec,
                query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                    condition=_query_expansion_condition(query_expansion)
                ),
                spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                    mode=_spell_correction_mode(spell_correction)
                ),
            )

            # page token if any
            if next_token:
                request.page_token = next_token

            # filter if provided
            if filter_expr:
                request.filter = filter_expr

            # optional search result mode
            # NOTE: enum 값은 클라이언트 버전에 따라 이름이 다를 수 있어
            #       안전하게 문자열로 받은 값을 매핑해서 설정
            if search_result_mode:
                mode = (search_result_mode or "").upper().strip()
                if mode == "CHUNKS":
                    request.content_search_spec.search_result_mode = (
                        discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
                    )
                elif mode == "DOCUMENTS":
                    request.content_search_spec.search_result_mode = (
                        discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.DOCUMENTS
                    )

            response = client.search(request=request)

            # ---- summary: only first page ----
            if pages_fetched == 1:
                summary_text = _extract_summary_text(response)
                summary_citations = _extract_summary_citations(response)

            # ---- parse results page ----
            for r in response.results:
                doc_dict = None
                if getattr(r, "document", None) is not None and hasattr(r.document, "to_dict"):
                    doc_dict = r.document.to_dict()

                raw_result = r.to_dict() if hasattr(r, "to_dict") else None

                title, uri, link = _doc_best_effort(doc_dict)
                extra = _extract_snippets_and_extractive(raw_result)

                if title is None and isinstance(doc_dict, dict):
                    title = _first_non_empty(doc_dict.get("name"), doc_dict.get("id"))
                if uri is None and isinstance(doc_dict, dict):
                    uri = _first_non_empty(
                        _get_in(doc_dict, ["content", "uri"]),
                        doc_dict.get("uri"),
                        doc_dict.get("link"),
                    )

                item = {
                    "id": getattr(r, "id", None),
                    "document": doc_dict,
                    "source": {"title": title, "uri": uri, "link": link},
                    "snippets": extra["snippets"],
                    "extractive_answers": extra["extractive_answers"],
                    "extractive_segments": extra["extractive_segments"],
                }
                results.append(item)

                # ---- budget accounting (chars) ----
                for s in item.get("snippets", []) or []:
                    if isinstance(s, str):
                        total_chars += len(s)

                for seg in item.get("extractive_segments", []) or []:
                    if isinstance(seg, dict):
                        txt = (
                            seg.get("content")
                            or seg.get("text")
                            or seg.get("snippet")
                            or ""
                        )
                        if isinstance(txt, str):
                            total_chars += len(txt)

                # ---- check result budget ----
                if len(results) >= max_total_results:
                    is_truncated = True
                    truncation_reason = "max_total_results_reached"
                    break

                # ---- check char budget ----
                if total_chars >= max_total_chars:
                    is_truncated = True
                    truncation_reason = "max_total_chars_exceeded"
                    break

            if is_truncated:
                break

            # ---- next page? ----
            next_token = getattr(response, "next_page_token", "") or ""
            if not next_token:
                break

            if pages_fetched >= max_pages:
                is_truncated = True
                truncation_reason = "max_pages_reached"
                break

        # 최소한 results 기반 sources 목록은 항상 만든다
        citations = _build_citation_items_from_results(results)

        # truncation 안내 메시지
        user_guidance = None
        if is_truncated:
            user_guidance = (
                "검색 범위(또는 결과)가 커서 모든 근거를 수집/요약하지 못했습니다. "
                "기간/키워드/조건을 더 좁혀서 다시 질문해 주세요."
            )

        return {
            "status": "success",
            "engine_id": engine_id,
            "serving_config": serving_config,
            "location": loc,
            "query": query_text,

            # summary (first page)
            "summary_text": summary_text,
            "summary_citations": summary_citations,

            # 사람이 읽을 수 있는 “Sources 리스트”
            "citations": citations,

            # results
            "results": results,
            "count": len(results),

            # paging/budget diagnostics
            "page_size": page_size_int,
            "pages_fetched": pages_fetched,
            "next_page_token": next_token if next_token else None,
            "is_truncated": is_truncated,
            "truncation_reason": truncation_reason,
            "total_chars_estimate": total_chars,

            "message": (
                user_guidance
                or f"Found {len(results)} result(s) for query: '{query_text}'"
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "engine_id": engine_id,
            "query": query_text,
            "error_message": str(e),
            "message": f"Failed to search Vertex AI Search: {str(e)}",
        }

# -------------------------
# Tool 2) list engines
# -------------------------

def list_search_engines(
    page_size: Optional[int] = None,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lists Vertex AI Search Apps (Engines) in the current project/location.
    """
    try:
        loc = _get_search_location()
        if page_size is None:
            page_size = 50

        parent = f"projects/{PROJECT_ID}/locations/{loc}/collections/default_collection"

        client = discoveryengine.EngineServiceClient(
            client_options=ClientOptions(api_endpoint=_discoveryengine_endpoint(loc))
        )

        request = discoveryengine.ListEnginesRequest(
            parent=parent,
            page_size=page_size,
            page_token=page_token or "",
        )

        resp = client.list_engines(request=request)

        engines = []
        for eng in resp.engines:
            eng_id = eng.name.split("/")[-1]
            engines.append(
                {
                    "id": eng_id,
                    "name": eng.name,
                    "display_name": getattr(eng, "display_name", None),
                    "industry_vertical": getattr(eng, "industry_vertical", None),
                    "solution_type": getattr(eng, "solution_type", None),
                }
            )

        return {
            "status": "success",
            "location": loc,
            "parent": parent,
            "engines": engines,
            "count": len(engines),
            "next_page_token": getattr(resp, "next_page_token", None),
            "message": f"Found {len(engines)} engine(s)",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "message": f"Failed to list engines: {str(e)}",
        }



# =========================
# Types
# =========================

DetectFn = Callable[[str], Any]
BuildFn = Callable[[Any], str]
StripFn = Callable[[str, Any], str]


@dataclass(frozen=True)
class FilterRule:
    field: str
    detect: DetectFn
    build: BuildFn
    strip: Optional[StripFn] = None


@dataclass(frozen=True)
class EngineConfig:
    key: str                  # 내부 식별자 (e.g., "work_hist")
    engine_id: str            # 실제 Discovery Engine ID
    patterns: List[str]       # 분류 키워드/패턴
    default_on_tie: bool      # 동점/애매 시 기본 선택 여부
    filter_rules: List[FilterRule]  # 지원 필터 규칙 (없으면 [])


# =========================
# Engine IDs (실제 엔진 ID)
# =========================

WORK_HIST_ENGINE_ID = "work-hist_1769394022215"
MANUAL_ENGINE_ID    = "manual_1769391381390"


# =========================
# Pattern sets (패턴 정하기)
# =========================

WORK_PATTERNS = [
    r"\b업무\b", r"\b업무일지\b", r"\b일지\b", r"\b작업\b", r"\b방문\b", r"\b점검\b",
    r"\b회의\b", r"\b미팅\b", r"\b일정\b", r"\b출장\b", r"\b리포트\b", r"\b보고서\b",
    r"\b거래처\b", r"\b담당자\b", r"\b장소\b", r"\b특이사항\b",
    r"\b202\d년\b", r"\b20\d{2}-\d{2}-\d{2}\b",
    r"\b\d{1,2}월\b",
]

MANUAL_PATTERNS = [
    r"\b매뉴얼\b", r"\b가이드\b", r"\b설명서\b", r"\b구성\b", r"\b설정\b", r"\b설치\b",
    r"\b업그레이드\b", r"\b버전\b", r"\b에러\b", r"\b오류\b", r"\b트러블슈팅\b",
    r"\bCLI\b", r"\b명령어\b", r"\b파라미터\b", r"\bRelease\b", r"\bKB\b",
    r"\bVMware\b", r"\bCisco\b", r"\bEMC\b", r"\bUnity\b", r"\bMDS\b",
]


# =========================
# Filter compiler helpers
# =========================
def _clean(s: str) -> str:
    return (s or "").strip()

def _month_range(y: int, m: int) -> Tuple[str, str]:
    last = calendar.monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"

def _detect_date(q: str) -> Optional[Dict[str, str]]:
    q = _clean(q)

    # 공통 ISO 날짜 패턴
    iso_pat = r"(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"

    # 1) YYYY-MM-DD ~ YYYY-MM-DD (기간)  ← 최우선
    m0 = re.search(
        rf"\b({iso_pat})\b\s*(?:~|∼|-|–|—|to|부터)\s*\b({iso_pat})\b(?:\s*(?:까지)?)",
        q,
        flags=re.IGNORECASE
    )
    if m0:
        start = m0.group(1)
        end = m0.group(5)   # 두번째 iso_pat의 전체 match 위치
        raw = m0.group(0)
        return {"type": "range", "start": start, "end": end, "raw": raw}

    # 2) YYYY-MM-DD (단일)
    m = re.search(rf"\b{iso_pat}\b", q)
    if m:
        iso = m.group(0)
        return {"type": "day", "start": iso, "end": iso, "raw": iso}

    # 3) YYYY년 M월 D일
    m1 = re.search(r"(20\d{2})\s*년\s*(1[0-2]|[1-9])\s*월\s*(3[01]|[12]\d|[1-9])\s*일", q)
    if m1:
        y = int(m1.group(1))
        mo = int(m1.group(2))
        d = int(m1.group(3))
        iso = f"{y:04d}-{mo:02d}-{d:02d}"
        return {"type": "day", "start": iso, "end": iso, "raw": m1.group(0)}

    # 4) YYYY년 M월
    m2 = re.search(r"(20\d{2})\s*년\s*(1[0-2]|[1-9])\s*월(?!\s*\d+\s*일)", q)
    if m2:
        y = int(m2.group(1))
        mo = int(m2.group(2))
        start, end = _month_range(y, mo)
        return {"type": "month", "start": start, "end": end, "raw": m2.group(0)}

    return None


def _build_date_filter(v: Dict[str, str]) -> str:
    return f'date >= "{v["start"]}" AND date <= "{v["end"]}"'

def _strip_date(q: str, v: Dict[str, str]) -> str:
    raw = v.get("raw", "")
    return _clean(q.replace(raw, "")) if raw else q

def _detect_owner(q: str) -> Optional[str]:
    m = re.search(r"(담당자|작성자)\s*[:：]\s*([가-힣A-Za-z0-9_]+)", q)
    return m.group(2).strip() if m else None

def _build_owner_filter(v: str) -> str:
    v = v.replace('"', '\\"')
    return f'owner: ANY("{v}")'

def _strip_owner(q: str, v: str) -> str:
    return _clean(re.sub(r"(담당자|작성자)\s*[:：]\s*" + re.escape(v), "", q))

def _detect_company(q: str) -> Optional[str]:
    m = re.search(r"(거래처|회사)\s*[:：]\s*([가-힣A-Za-z0-9_()\-]+)", q)
    return m.group(2).strip() if m else None

def _build_company_filter(v: str) -> str:
    v = v.replace('"', '\\"')
    return f'company: ANY("{v}")'

def _strip_company(q: str, v: str) -> str:
    return _clean(re.sub(r"(거래처|회사)\s*[:：]\s*" + re.escape(v), "", q))


# =========================
# Engine Registry (추가시 여기에 확장)
# =========================

ENGINE_REGISTRY: List[EngineConfig] = [
    EngineConfig(
        key="work_hist",
        engine_id=WORK_HIST_ENGINE_ID,
        patterns=WORK_PATTERNS,
        default_on_tie=True,  # 애매하면 work_hist
        filter_rules=[
            FilterRule("date", _detect_date, _build_date_filter, _strip_date),
            FilterRule("owner", _detect_owner, _build_owner_filter, _strip_owner),
            FilterRule("company", _detect_company, _build_company_filter, _strip_company),
        ],
    ),
    EngineConfig(
        key="manual",
        engine_id=MANUAL_ENGINE_ID,
        patterns=MANUAL_PATTERNS,
        default_on_tie=False,
        filter_rules=[],  # 현재는 manual엔진 필터 없음 (추가되면 여기만 넣으면 됨)
    ),
]


# =========================
# Core: select + compile
# =========================

def select_and_compile(user_query: str) -> Dict[str, Any]:
    """
    1) 엔진을 선택한다 (패턴 점수 기반)
    2) 선택된 엔진의 filter_rules가 있으면 filter_expr + query_text를 컴파일한다
    3) 항상 동일한 스키마로 반환한다

    반환 스키마:
    {
      "engine_id": str,
      "engine_key": str,
      "engine_reason": str,
      "scores": {engine_key: int, ...},
      "matched_patterns": {engine_key: [pattern, ...], ...},
      "compiled": {
         "applied": bool,
         "filter_expr": Optional[str],
         "query_text": str,
         "reason": str,
         "matched_fields": {field: value, ...}
      }
    }
    """
    q = _clean(user_query)
    q_for_match = q.lower()

    scores: Dict[str, int] = {}
    matched_patterns: Dict[str, List[str]] = {}

    # (1) score 계산
    for cfg in ENGINE_REGISTRY:
        s = 0
        hits: List[str] = []
        for p in cfg.patterns:
            if re.search(p, q_for_match, flags=re.IGNORECASE):
                s += 1
                hits.append(p)
        scores[cfg.key] = s
        matched_patterns[cfg.key] = hits

    # (2) 최고점 엔진 선택 (+ tie-break)
    best_score = max(scores.values()) if scores else 0
    candidates = [cfg for cfg in ENGINE_REGISTRY if scores.get(cfg.key, 0) == best_score]

    if len(candidates) == 1:
        selected = candidates[0]
        engine_reason = f"{selected.key} 점수가 가장 높음 ({selected.key}={best_score})"
    else:
        # 동점: default_on_tie가 True인 엔진이 있으면 그걸 선택, 없으면 레지스트리 첫번째
        default_candidates = [c for c in candidates if c.default_on_tie]
        selected = default_candidates[0] if default_candidates else candidates[0]
        engine_reason = (
            f"동점({best_score}) → default_on_tie 규칙으로 {selected.key} 선택"
            if default_candidates
            else f"동점({best_score}) → 우선순위(레지스트리 순서)로 {selected.key} 선택"
        )

    # (3) 선택된 엔진에 대해 필터 컴파일
    compiled = _compile_with_rules(selected, q)

    return {
        "engine_id": selected.engine_id,
        "engine_key": selected.key,
        "engine_reason": engine_reason,
        "scores": scores,
        "matched_patterns": matched_patterns,
        "compiled": compiled,
    }


def _compile_with_rules(cfg: EngineConfig, user_query: str) -> Dict[str, Any]:
    q = _clean(user_query)

    # 필터 규칙이 없으면 그대로 반환 (확장 가능)
    if not cfg.filter_rules:
        return {
            "applied": False,
            "filter_expr": None,
            "query_text": q,
            "reason": "engine_has_no_filter_rules",
            "matched_fields": {},
        }

    filters: List[str] = []
    matched_fields: Dict[str, Any] = {}
    query_text = q

    # NOTE: detect는 원문(q) 기반, strip은 누적된 query_text에 적용
    for rule in cfg.filter_rules:
        val = rule.detect(q)
        if val is None or val == "":
            continue

        matched_fields[rule.field] = val
        filters.append(rule.build(val))

        if rule.strip:
            query_text = rule.strip(query_text, val)

    filter_expr = None
    if filters:
        chunks = [f"({f})" if " AND " in f else f for f in filters]
        filter_expr = " AND ".join(chunks)

    return {
        "applied": bool(filter_expr),
        "filter_expr": filter_expr,
        "query_text": query_text if query_text else q,
        "reason": "compiled_from_engine_rules" if filter_expr else "no_fields_matched",
        "matched_fields": matched_fields,
    }

# -------------------------
# ADK FunctionTools
# -------------------------

vertex_search_tool = FunctionTool(vertex_search)
list_search_engines_tool = FunctionTool(list_search_engines)
select_and_compile = FunctionTool(select_and_compile)