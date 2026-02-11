"""
Minimal interactive CLI for the ACE-Enhanced LangGraph Agent.

Usage:
    python cli.py                         # Interactive REPL mode
    python cli.py --query "What is 2+2?"  # Single query mode
    python cli.py --learner <id>          # Specify learner context
"""

from __future__ import annotations

import argparse
import sys
import time
import json

from dotenv import load_dotenv
load_dotenv()

from src.agent import build_ace_graph, get_ace_system
from src.llm import get_metrics_collector


def run_query(app, query: str, learner_id: str = "default_user", context_scope_id: str = None):
    cfg = {"configurable": {"thread_id": f"cli-{int(time.time() * 1000)}"}}

    scratch = {
        "ace_online_learning": True,
        "learner_id": learner_id,
    }
    if context_scope_id:
        scratch["context_scope_id"] = context_scope_id

    state = {
        "messages": [{"role": "user", "content": query}],
        "mode": "",
        "scratch": scratch,
        "result": {},
    }

    output = app.invoke(state, config=cfg)

    answer = output.get("result", {}).get("answer", "(no answer)")
    mode = output.get("mode", "unknown")
    ace_delta = output.get("scratch", {}).get("ace_delta")

    collector = get_metrics_collector()
    summary = collector.summary()

    return {
        "answer": answer,
        "mode": mode,
        "ace_delta": ace_delta,
        "metrics": summary,
    }


def interactive_mode(learner_id: str, context_scope_id: str = None):
    app = build_ace_graph()

    print("=" * 60)
    print("  ACE-Enhanced LangGraph Agent (Interactive CLI)")
    print("=" * 60)
    print(f"  Learner: {learner_id}")
    if context_scope_id:
        print(f"  Context Scope: {context_scope_id}")
    print("  Type 'quit' to exit, 'stats' for memory statistics")
    print("=" * 60)

    while True:
        try:
            query = input("\n> ").strip()
            if not query:
                continue
            if query.lower() in {"quit", "exit"}:
                break
            if query.lower() == "stats":
                memory, _ = get_ace_system(learner_id, context_scope_id=context_scope_id)
                stats = memory.get_statistics()
                print(f"\nMemory Statistics:")
                print(f"  Total bullets: {stats['total_bullets']}")
                if stats["total_bullets"] > 0:
                    print(f"  Avg score: {stats['avg_score']:.3f}")
                    if stats.get("categories"):
                        for tag, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)[:10]:
                            print(f"    {tag}: {count}")
                continue
            if query.lower() == "metrics":
                collector = get_metrics_collector()
                summary = collector.summary()
                print(f"\nMetrics Summary:")
                print(json.dumps(summary, indent=2))
                continue

            result = run_query(app, query, learner_id=learner_id, context_scope_id=context_scope_id)
            print(f"\nAnswer: {result['answer']}")
            print(f"Mode: {result['mode']}")
            if result.get("ace_delta"):
                d = result["ace_delta"]
                print(f"ACE Delta: +{d.get('num_new_bullets', 0)} new, {d.get('num_updates', 0)} updates, {d.get('num_removals', 0)} removals")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="ACE Agent CLI")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--learner", type=str, default="default_user", help="Learner ID")
    parser.add_argument("--context-scope", type=str, default=None, help="Context scope ID for memory isolation")
    args = parser.parse_args()

    if args.query:
        app = build_ace_graph()
        result = run_query(app, args.query, learner_id=args.learner, context_scope_id=args.context_scope)
        print(json.dumps(result, indent=2, default=str))
    else:
        interactive_mode(args.learner, context_scope_id=args.context_scope)


if __name__ == "__main__":
    main()
