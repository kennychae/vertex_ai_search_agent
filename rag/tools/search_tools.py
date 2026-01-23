"""
Tools for Vertex AI Search (Discovery Engine) using ADK FunctionTool pattern.

- vertex_search(): returns search results + snippets + extractive answers/segments + summary (when available)
- list_search_engines(): lists available Vertex AI Search apps (engines)
"""

from typing import Dict, Any, Optional, List, Tuple

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
    ✅ 예전처럼 SearchResponse.summary에서 바로 요약 텍스트를 뽑는다.
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
    ✅ summary citations가 없을 때도 "Sources 목록"을 만들 수 있게 results의 source로 citations 생성.
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
    ✅ summary_text는 예전처럼 resp.summary에서 뽑는다(가장 안정적).
    ✅ citations는 (1) summary citations가 있으면 그것도 내려주고,
                 (2) 최소한 results에서 sources 목록을 만들어 항상 제공한다.
    """
    try:
        # ---- defaults ----
        if serving_config_id is None:
            serving_config_id = SEARCH_DEFAULT_SERVING_CONFIG_ID
        if page_size is None:
            page_size = SEARCH_DEFAULT_PAGE_SIZE

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

        # ---- client / request ----
        loc = _get_search_location()
        serving_config = _build_serving_config(engine_id, serving_config_id)

        client = discoveryengine.SearchServiceClient(
            client_options=ClientOptions(api_endpoint=_discoveryengine_endpoint(loc))
        )

        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
                max_snippet_count=max_snippet_count
            ),
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=max_extractive_answer_count,
                max_extractive_segment_count=max_extractive_segment_count,
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

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query_text,
            page_size=page_size,
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=_query_expansion_condition(query_expansion)
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=_spell_correction_mode(spell_correction)
            ),
        )
        if filter_expr:
            request.filter = filter_expr

        response = client.search(request=request)

        # ---- parse results ----
        results: List[Dict[str, Any]] = []
        for r in response.results:
            doc_dict = None
            if getattr(r, "document", None) is not None and hasattr(r.document, "to_dict"):
                doc_dict = r.document.to_dict()

            # raw_result는 extractive/snippet 파싱용으로만 쓰고 반환에는 포함하지 않음(크기 방지)
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

            results.append(
                {
                    "id": getattr(r, "id", None),
                    "document": doc_dict,
                    "source": {"title": title, "uri": uri, "link": link},
                    "snippets": extra["snippets"],
                    "extractive_answers": extra["extractive_answers"],
                    "extractive_segments": extra["extractive_segments"],
                }
            )

        # ✅ summary는 예전처럼 summary 필드에서 뽑는다
        summary_text = _extract_summary_text(response)
        summary_citations = _extract_summary_citations(response)

        # ✅ 최소한 results 기반 sources 목록은 항상 만든다
        citations = _build_citation_items_from_results(results)

        return {
            "status": "success",
            "engine_id": engine_id,
            "serving_config": serving_config,
            "location": loc,
            "query": query_text,

            "summary_text": summary_text,
            "summary_citations": summary_citations,

            # 사람이 읽을 수 있는 “Sources 리스트”
            "citations": citations,

            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} result(s) for query: '{query_text}'",
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


# -------------------------
# ADK FunctionTools
# -------------------------

vertex_search_tool = FunctionTool(vertex_search)
list_search_engines_tool = FunctionTool(list_search_engines)
