"""
Evaluation script — runs the golden dataset (data/eval_dataset.json)
through the full agent graph and reports:
  - route accuracy       (did the router pick the expected branch?)
  - keyword coverage     (does the answer contain the expected facts?)
  - groundedness rate    (did the validator mark the answer as grounded?)
  - abstention accuracy  (for out-of-scope questions, did it correctly
                           say it doesn't have enough information, rather
                           than hallucinating an answer?)

Run:
    python -m evaluation.run_evaluation

Prerequisite: ingest the sample policy doc first (from repo root):
    python -c "
from src.services.vector_store import ingest_document
ingest_document(open('data/company_policies.txt').read(), source='company_policies')
"
"""
import json
from pathlib import Path

from src.agents.graph import compiled_graph

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_dataset.json"

# Phrases that indicate the model correctly abstained rather than
# hallucinating an answer for an out-of-scope question.
_ABSTENTION_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "doesn't contain", "does not contain",
    "no information", "not mentioned", "not covered",
    "cannot find", "can't find", "unable to find",
    "not specified", "not provided",
]


def _is_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in _ABSTENTION_MARKERS)


def run_evaluation() -> None:
    cases = json.loads(EVAL_FILE.read_text())

    route_correct = 0
    keyword_hits = 0
    keyword_total = 0
    grounded_count = 0
    abstention_cases = [c for c in cases if c.get("expect_no_answer")]
    abstention_correct = 0

    print(f"Running {len(cases)} evaluation cases "
          f"({len(abstention_cases)} abstention tests)...\n")

    for i, case in enumerate(cases, start=1):
        result = compiled_graph.invoke({"user_query": case["query"], "retry_count": 0})
        answer = (result.get("final_answer") or result.get("draft_answer", "")).lower()
        route = result.get("route", "")
        is_grounded = result.get("is_grounded", False)

        route_ok = route == case["expected_route"]
        route_correct += int(route_ok)
        grounded_count += int(is_grounded)

        if case.get("expect_no_answer"):
            abstained = _is_abstention(answer)
            abstention_correct += int(abstained)
            status = "✓" if abstained else "✗"
            print(f"[{status}] Case {i} (abstention test): \"{case['query']}\"")
            print(f"      abstained={abstained} route={route} grounded={is_grounded}")
            print(f"      answer: {answer[:150]}{'...' if len(answer) > 150 else ''}\n")
            continue

        expected_keywords = case.get("expected_answer_contains", [])
        case_hit = False
        for kw in expected_keywords:
            keyword_total += 1
            if kw.lower() in answer:
                keyword_hits += 1
                case_hit = True

        status = "✓" if route_ok and (not expected_keywords or case_hit) else "✗"
        print(f"[{status}] Case {i}: \"{case['query']}\"")
        print(f"      route={route} (expected={case['expected_route']}) grounded={is_grounded}")
        print(f"      answer: {answer[:150]}{'...' if len(answer) > 150 else ''}\n")

    n = len(cases)
    print("── Summary " + "─" * 40)
    print(f"Route accuracy:       {route_correct}/{n} ({100 * route_correct / n:.1f}%)")
    if keyword_total:
        print(f"Keyword coverage:     {keyword_hits}/{keyword_total} ({100 * keyword_hits / keyword_total:.1f}%)")
    print(f"Groundedness rate:    {grounded_count}/{n} ({100 * grounded_count / n:.1f}%)")
    if abstention_cases:
        print(f"Abstention accuracy:  {abstention_correct}/{len(abstention_cases)} "
              f"({100 * abstention_correct / len(abstention_cases):.1f}%)  "
              f"— correctly said 'I don't know' on out-of-scope questions")


if __name__ == "__main__":
    run_evaluation()