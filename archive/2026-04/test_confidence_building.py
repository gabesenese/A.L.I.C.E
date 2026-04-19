"""
Extended Alice Confidence Building Test
========================================
Run many interactions to build Alice's confidence and independence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import ALICE
import time

def build_confidence():
    """Run extensive interactions to build Alice's confidence"""

    print("=" * 70)
    print("ALICE CONFIDENCE BUILDING - EXTENDED TEST")
    print("=" * 70)
    print()

    # Initialize Alice
    print("Initializing Alice...")
    alice = ALICE(llm_model='llama3.1:8b', voice_enabled=False)
    print()

    # Define test scenarios - each will be repeated to build confidence
    scenarios = [
        # Identity questions (repeat 5x to build confidence)
        ("Who created you?", 5),
        ("What is your name?", 5),
        ("Who are you?", 5),

        # Factual learning (repeat to strengthen entities)
        ("My name is Gabriel", 3),
        ("I live in Waterloo", 3),
        ("I am a software developer", 3),
        ("I work on AI projects", 3),

        # Knowledge recall questions
        ("Where do I live?", 4),
        ("What do I do for work?", 4),
        ("What is my name?", 4),

        # Relationship building
        ("You help me with coding", 2),
        ("You are my AI assistant", 2),
        ("We work together on projects", 2),

        # General conversation
        ("How are you today?", 3),
        ("What can you help me with?", 3),
        ("Tell me about yourself", 3),
    ]

    print("Running confidence-building scenarios...")
    print("This will take several minutes...")
    print()

    total_interactions = sum(count for _, count in scenarios)
    current = 0

    for question, repeat_count in scenarios:
        print(f"\nScenario: '{question}' (x{repeat_count})")
        print("-" * 70)

        for i in range(repeat_count):
            current += 1
            progress = (current / total_interactions) * 100
            print(f"[{progress:5.1f}%] Iteration {i+1}/{repeat_count}...", end=" ")

            response = alice.process_input(question)

            # Check confidence
            can_answer, conf = alice.knowledge_engine.can_answer_independently(
                question, "conversation:question"
            )

            print(f"Confidence: {conf:.2f} {'[CAN ANSWER INDEPENDENTLY]' if can_answer else ''}")

            # Small delay to not overwhelm Ollama
            if i < repeat_count - 1:
                time.sleep(0.5)

        # Show what Alice learned from this scenario
        stats = alice.knowledge_engine.get_stats()
        print(f"  Entities: {stats['total_entities']}, Relationships: {stats['total_relationships']}")

    print()
    print("=" * 70)
    print("CONFIDENCE BUILDING COMPLETE")
    print("=" * 70)
    print()

    # Final statistics
    stats = alice.knowledge_engine.get_stats()
    print(f"Total Entities Learned: {stats['total_entities']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Topics with High Confidence (>0.7): {stats['topics_confident_in']}")
    print()

    # Show top entities
    print("Top Mentioned Entities:")
    for entity, count in stats['top_entities'][:10]:
        entity_info = alice.knowledge_engine.get_entity_info(entity)
        if entity_info:
            print(f"  - {entity}: {count} mentions, confidence: {entity_info['confidence']:.2f}")
    print()

    # Show topic confidence levels
    print("Topic Confidence Levels:")
    for topic, confidence in sorted(
        alice.knowledge_engine.topic_confidence.items(),
        key=lambda x: x[1], reverse=True
    )[:15]:
        bar = "█" * int(confidence * 50)
        status = "[INDEPENDENT]" if confidence > 0.7 else ""
        print(f"  {topic:.<30} {bar} {confidence:.2f} {status}")
    print()

    # Test independence on common questions
    print("=" * 70)
    print("INDEPENDENCE TEST - Can Alice Answer Without Ollama?")
    print("=" * 70)
    print()

    test_questions = [
        "Who created you?",
        "What is your name?",
        "Where do I live?",
        "What do I do?",
        "Who are you?"
    ]

    for question in test_questions:
        can_answer, conf = alice.knowledge_engine.can_answer_independently(
            question, "conversation:question"
        )
        status = "YES - Independent" if can_answer else "NO - Needs Ollama"
        print(f"{question:.<40} {conf:.2f} [{status}]")

    print()
    print("Knowledge saved to data/knowledge/")
    print()

if __name__ == "__main__":
    build_confidence()
