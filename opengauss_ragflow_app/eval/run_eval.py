import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def evaluate_retrieval(base_url: str, item: dict) -> tuple[dict, float]:
    payload = {
        "vector_db_id": item["vector_db_id"],
        "query": item["question"],
        "mode": item.get("mode", "hybrid"),
        "query_rewrite": item.get("query_rewrite", False),
        "max_chunks": item.get("max_chunks", 5),
        "ranker_type": item.get("ranker_type", "rrf"),
        "alpha": item.get("alpha", 0.6),
        "impact_factor": item.get("impact_factor", 60),
    }
    start = time.perf_counter()
    response = post_json(base_url, "/api/retrieval/query", payload)
    latency_ms = (time.perf_counter() - start) * 1000

    gold_ids = set(item.get("gold_document_ids", []))
    ranked_ids = [chunk.get("document_id") for chunk in response.get("chunks", []) if chunk.get("document_id")]
    hit = any(doc_id in gold_ids for doc_id in ranked_ids)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": hit,
        "mrr": reciprocal_rank,
        "ranked_ids": ranked_ids,
        "raw": response,
    }, latency_ms


def evaluate_chat(base_url: str, item: dict) -> tuple[dict, float]:
    session_payload = {
        "vector_db_id": item["vector_db_id"],
        "model_id": item.get("model_id"),
        "mode": item.get("mode", "hybrid"),
        "query_rewrite": item.get("query_rewrite", False),
        "max_chunks": item.get("max_chunks", 5),
        "ranker_type": item.get("ranker_type", "rrf"),
        "alpha": item.get("alpha", 0.6),
        "impact_factor": item.get("impact_factor", 60),
        "instructions": item.get("instructions", "Answer using retrieved context only."),
    }
    session = post_json(base_url, "/api/chat/sessions", session_payload)

    start = time.perf_counter()
    response = post_json(
        base_url,
        f"/api/chat/sessions/{session['session_id']}/messages",
        {"message": item["question"]},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    gold_ids = set(item.get("gold_document_ids", []))
    citation_ids = {citation.get("document_id") for citation in response.get("citations", []) if citation.get("document_id")}
    citation_hit = bool(gold_ids & citation_ids) if gold_ids else False

    gold_answer = item.get("gold_answer")
    exact_match = None
    if gold_answer:
        exact_match = normalize_text(response.get("answer", "")) == normalize_text(gold_answer)

    return {
        "citation_hit": citation_hit,
        "exact_match": exact_match,
        "raw": response,
    }, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--app-base-url", default="http://localhost:8787")
    parser.add_argument("--task", choices=["retrieval", "chat", "both"], default="both")
    parser.add_argument("--details-out")
    args = parser.parse_args()

    items = load_jsonl(Path(args.dataset))
    retrieval_hits = 0
    retrieval_mrr_total = 0.0
    retrieval_latency_total = 0.0
    chat_latency_total = 0.0
    chat_exact_total = 0
    chat_exact_count = 0
    citation_hits = 0
    details: list[dict] = []
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "retrieval_hits": 0.0,
            "retrieval_mrr": 0.0,
            "citation_hits": 0.0,
            "chat_exact_total": 0.0,
            "chat_exact_count": 0.0,
        }
    )

    for item in items:
        question_type = item.get("question_type", "default")
        grouped[question_type]["count"] += 1
        print(f"[eval] {item.get('id', item['question'])}")
        row = {
            "id": item.get("id", item["question"]),
            "question_type": question_type,
            "question": item["question"],
        }
        if args.task in {"retrieval", "both"}:
            result, latency_ms = evaluate_retrieval(args.app_base_url, item)
            retrieval_hits += int(result["hit"])
            retrieval_mrr_total += result["mrr"]
            retrieval_latency_total += latency_ms
            grouped[question_type]["retrieval_hits"] += int(result["hit"])
            grouped[question_type]["retrieval_mrr"] += result["mrr"]
            row["retrieval"] = {
                "hit": result["hit"],
                "mrr": result["mrr"],
                "latency_ms": latency_ms,
                "ranked_ids": result["ranked_ids"],
            }
            print(f"  retrieval_hit={result['hit']} mrr={result['mrr']:.4f} latency_ms={latency_ms:.2f}")

        if args.task in {"chat", "both"}:
            result, latency_ms = evaluate_chat(args.app_base_url, item)
            chat_latency_total += latency_ms
            citation_hits += int(result["citation_hit"])
            grouped[question_type]["citation_hits"] += int(result["citation_hit"])
            if result["exact_match"] is not None:
                chat_exact_count += 1
                chat_exact_total += int(result["exact_match"])
                grouped[question_type]["chat_exact_total"] += int(result["exact_match"])
                grouped[question_type]["chat_exact_count"] += 1
            row["chat"] = {
                "citation_hit": result["citation_hit"],
                "exact_match": result["exact_match"],
                "latency_ms": latency_ms,
                "answer": result["raw"].get("answer"),
                "citations": result["raw"].get("citations"),
            }
            print(
                f"  citation_hit={result['citation_hit']} "
                f"exact_match={result['exact_match']} latency_ms={latency_ms:.2f}"
            )
        details.append(row)

    count = max(len(items), 1)
    summary = {
        "samples": len(items),
        "retrieval_recall_at_k": retrieval_hits / count if args.task in {"retrieval", "both"} else None,
        "retrieval_mrr": retrieval_mrr_total / count if args.task in {"retrieval", "both"} else None,
        "avg_retrieval_latency_ms": retrieval_latency_total / count if args.task in {"retrieval", "both"} else None,
        "chat_exact_match": (chat_exact_total / chat_exact_count) if chat_exact_count else None,
        "chat_citation_hit_rate": citation_hits / count if args.task in {"chat", "both"} else None,
        "avg_chat_latency_ms": chat_latency_total / count if args.task in {"chat", "both"} else None,
    }
    by_type = {}
    for question_type, stats in grouped.items():
        bucket_count = max(int(stats["count"]), 1)
        by_type[question_type] = {
            "samples": int(stats["count"]),
            "retrieval_recall_at_k": stats["retrieval_hits"] / bucket_count if args.task in {"retrieval", "both"} else None,
            "retrieval_mrr": stats["retrieval_mrr"] / bucket_count if args.task in {"retrieval", "both"} else None,
            "chat_citation_hit_rate": stats["citation_hits"] / bucket_count if args.task in {"chat", "both"} else None,
            "chat_exact_match": (
                stats["chat_exact_total"] / stats["chat_exact_count"] if stats["chat_exact_count"] else None
            ),
        }

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== by_question_type ===")
    print(json.dumps(by_type, ensure_ascii=False, indent=2))

    if args.details_out:
        Path(args.details_out).write_text(
            json.dumps({"summary": summary, "by_question_type": by_type, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc
