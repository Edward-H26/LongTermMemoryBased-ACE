"""
ACE Memory Analysis Tool

Inspect, search, and export ACE memory for a specific learner.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from src.ace_memory import ACEMemory, Bullet
from src.storage import Neo4jMemoryStore


def _load_memory(learner_id: str) -> ACEMemory:
    if not learner_id:
        raise ValueError("Learner ID is required. Use --learner or set ACE_ANALYZE_LEARNER_ID.")
    store = Neo4jMemoryStore(learner_id)
    return ACEMemory(storage=store)


def analyze_memory(learner_id: str) -> None:
    memory = _load_memory(learner_id)
    if not memory.bullets:
        print("Memory is empty. Run the agent to accumulate strategies.")
        return

    stats = memory.get_statistics()
    print(f"\nTotal bullets: {stats['total_bullets']}")
    print(f"Average score: {stats['avg_score']:.3f}")
    print(f"Average helpful count: {stats['avg_helpful']:.1f}")
    print(f"Average harmful count: {stats['avg_harmful']:.1f}")

    if stats.get("categories"):
        print("\nCategories:")
        for tag, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {count} bullets")

    print("\nTop 10 Performing Bullets:")
    bullets_list = sorted(memory.bullets.values(), key=lambda b: (b.score(), b.helpful_count), reverse=True)
    for i, bullet in enumerate(bullets_list[:10], 1):
        print(f"  {i}. {bullet.format_for_prompt()}")
        print(f"     ID: {bullet.id} | Tags: {', '.join(bullet.tags)}")


def search_memory(query: str, learner_id: str, top_k: int = 10) -> None:
    memory = _load_memory(learner_id)
    if not memory.bullets:
        print("Memory is empty.")
        return

    bullets = memory.retrieve_relevant_bullets(query, top_k=top_k)
    if not bullets:
        print("No relevant bullets found.")
        return

    print(f"\nFound {len(bullets)} relevant bullets:")
    for i, bullet in enumerate(bullets, 1):
        print(f"  {i}. {bullet.format_for_prompt()}")
        print(f"     ID: {bullet.id} | Tags: {', '.join(bullet.tags)}")


def export_memory(learner_id: str, output_file: str = "ace_memory_export.txt") -> None:
    memory = _load_memory(learner_id)
    if not memory.bullets:
        print("Memory is empty.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ACE MEMORY EXPORT\n")
        f.write("=" * 80 + "\n\n")
        stats = memory.get_statistics()
        f.write(f"Total Bullets: {stats['total_bullets']}\n")
        f.write(f"Average Score: {stats['avg_score']:.3f}\n\n")

        bullets_list = sorted(memory.bullets.values(), key=lambda b: (b.score(), b.helpful_count), reverse=True)
        for i, bullet in enumerate(bullets_list, 1):
            f.write(f"{i}. {bullet.format_for_prompt()}\n")
            f.write(f"   ID: {bullet.id} | Score: {bullet.score():.3f} | Tags: {', '.join(bullet.tags)}\n\n")

    print(f"Memory exported to: {output_file}")


def interactive_mode(learner_id: str) -> None:
    print("=" * 60)
    print("  ACE Memory Analysis - Interactive Mode")
    print("=" * 60)
    print("  Commands: stats, top, recent, search <query>, export, quit")
    print()

    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue
            if command in {"quit", "exit"}:
                break
            if command == "stats":
                analyze_memory(learner_id)
            elif command == "top":
                memory = _load_memory(learner_id)
                bullets_list = sorted(memory.bullets.values(), key=lambda b: (b.score(), b.helpful_count), reverse=True)[:10]
                for i, bullet in enumerate(bullets_list, 1):
                    print(f"  {i}. {bullet.format_for_prompt()}")
            elif command.startswith("search "):
                search_memory(command[7:].strip(), learner_id)
            elif command == "export":
                export_memory(learner_id)
            else:
                print("Unknown command.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as exc:
            print(f"Error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Inspect ACE memory stored in Neo4j")
    parser.add_argument("command", nargs="?", default="analyze", choices=["analyze", "search", "export", "interactive"])
    parser.add_argument("query", nargs="*", help="Query text for search command")
    parser.add_argument("--learner", default="default_user", dest="learner_id")
    args = parser.parse_args()

    learner_id = args.learner_id or os.getenv("ACE_ANALYZE_LEARNER_ID", "default_user")

    if args.command == "analyze":
        analyze_memory(learner_id)
    elif args.command == "search":
        if not args.query:
            print("Usage: analyze_memory.py search <query> --learner <id>")
        else:
            search_memory(" ".join(args.query), learner_id)
    elif args.command == "export":
        export_memory(learner_id)
    elif args.command == "interactive":
        interactive_mode(learner_id)


if __name__ == "__main__":
    main()
