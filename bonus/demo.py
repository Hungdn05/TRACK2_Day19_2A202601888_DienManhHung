"""
bonus/demo.py — HybridMemoryAgent Demo

5 queries demonstrating different memory retrieval scenarios:
1. Simple factual recall (vector hit)
2. Requires user profile context (topic_affinity)
3. Requires recent activity (queries_last_hour)
4. Paraphrase query (vector search wins)
5. Mixed query (hybrid + profile)

Run: python bonus/demo.py
Exit: 0 on success
"""

import sys
sys.path.insert(0, ".")

from bonus.agent import HybridMemoryAgent


def main():
    print("=" * 70)
    print("HybridMemoryAgent Demo — 5 Query Scenarios")
    print("=" * 70)

    agent = HybridMemoryAgent()

    # ── Seed memories for user ─────────────────────────────────────────────
    print("\n[Setup] Seeding memories for user u_001...")
    agent.remember(
        "User asked: 'How do I set up Kubernetes autoscaling with HPA?'",
        user_id="u_001",
        source="conversation",
    )
    agent.remember(
        "User read: 'Cost optimization in cloud: using spot instances and autoscaling'",
        user_id="u_001",
        source="document",
    )
    agent.remember(
        "User asked: 'Best practices for CI/CD with GitHub Actions'",
        user_id="u_001",
        source="conversation",
    )
    agent.remember(
        "User asked: 'How to configure network security groups in Azure?'",
        user_id="u_001",
        source="conversation",
    )
    agent.remember(
        "User read: 'Introduction to machine learning model deployment'",
        user_id="u_001",
        source="document",
    )
    print("[Setup] Done. 5 memories seeded.\n")

    # ── Query 1: Simple factual recall ─────────────────────────────────────
    print("-" * 70)
    print("QUERY 1: Simple factual recall (just vector hit)")
    print('         "Tôi đã hỏi gì về Kubernetes?"')
    print("-" * 70)
    context1 = agent.recall("Tôi đã hỏi gì về Kubernetes?", user_id="u_001")
    print(f"Context: {context1}\n")

    # ── Query 2: Needs profile context ─────────────────────────────────────
    print("-" * 70)
    print("QUERY 2: Needs user profile context (topic_affinity)")
    print('         "Recommend tôi đọc gì tiếp theo?"')
    print("-" * 70)
    context2 = agent.recall(
        "Recommend tôi đọc gì tiếp theo?",
        user_id="u_001",
    )
    print(f"Context: {context2}\n")

    # ── Query 3: Needs recent activity ─────────────────────────────────────
    print("-" * 70)
    print("QUERY 3: Needs fresh activity (queries_last_hour)")
    print('         "Tôi đang quan tâm gì gần đây?"')
    print("-" * 70)
    context3 = agent.recall(
        "Tôi đang quan tâm gì gần đây?",
        user_id="u_001",
    )
    print(f"Context: {context3}\n")

    # ── Query 4: Paraphrase (vector wins) ─────────────────────────────────
    print("-" * 70)
    print("QUERY 4: Paraphrase — semantic search wins")
    print('         "Tài liệu về tự động mở rộng hạ tầng?"')
    print("-" * 70)
    context4 = agent.recall(
        "Tài liệu về tự động mở rộng hạ tầng?",
        user_id="u_001",
    )
    print(f"Context: {context4}\n")

    # ── Query 5: Mixed (hybrid + profile) ─────────────────────────────────
    print("-" * 70)
    print("QUERY 5: Mixed — hybrid + user profile")
    print('         "Summary về cloud security"')
    print("-" * 70)
    context5 = agent.recall(
        "Summary về cloud security cho tôi",
        user_id="u_001",
    )
    print(f"Context: {context5}\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — All 5 queries executed successfully")
    print("=" * 70)
    print("""
Query # | Type                  | What it demonstrates
--------|----------------------|--------------------------------------------
   1    | Simple recall        | Vector search returns relevant memories
   2    | Profile context      | User topic_affinity affects context
   3    | Recent activity      | queries_last_hour from Feast feature store
   4    | Paraphrase           | Semantic search finds related docs
   5    | Mixed + Profile      | Combines episodic + stable profile
""")
    print("✅ Demo completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
