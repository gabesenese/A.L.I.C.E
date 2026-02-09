"""
Test Alice's Communication and Knowledge Learning
================================================
This script tests Alice's:
1. Basic communication
2. Knowledge learning and retention
3. Entity/relationship learning
4. Progressive independence
5. Confidence growth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import ALICE

def test_alice_knowledge():
    """Test Alice's knowledge engine and learning"""

    print("=" * 70)
    print("ALICE KNOWLEDGE & COMMUNICATION TEST")
    print("=" * 70)
    print()

    # Initialize Alice
    print("Initializing Alice...")
    alice = ALICE(llm_model='llama3.1:8b', voice_enabled=False)
    print()

    # Test 1: Basic communication
    print("TEST 1: Basic Communication")
    print("-" * 70)
    response = alice.process_input("Hello Alice, how are you?")
    print(f"User: Hello Alice, how are you?")
    print(f"Alice: {response}")
    print()

    # Test 2: Teach Alice a fact (entity learning)
    print("TEST 2: Teaching Alice Facts (Entity Learning)")
    print("-" * 70)
    response = alice.process_input("My name is Gabriel and I live in Waterloo")
    print(f"User: My name is Gabriel and I live in Waterloo")
    print(f"Alice: {response}")
    print()

    # Check what Alice learned
    print("What Alice learned:")
    gabriel_info = alice.knowledge_engine.get_entity_info("Gabriel")
    if gabriel_info:
        print(f"  - Entity: Gabriel ({gabriel_info['type']})")
        print(f"  - Confidence: {gabriel_info['confidence']:.2f}")
        print(f"  - Mentions: {gabriel_info['mention_count']}")

    waterloo_info = alice.knowledge_engine.get_entity_info("Waterloo")
    if waterloo_info:
        print(f"  - Entity: Waterloo ({waterloo_info['type']})")
        print(f"  - Confidence: {waterloo_info['confidence']:.2f}")

    # Check relationships
    relationships = alice.knowledge_engine.get_relationships_for_entity("Gabriel")
    if relationships:
        print(f"  - Relationships: {len(relationships)} found")
        for rel in relationships[:3]:
            print(f"    * {rel['subject']} {rel['predicate']} {rel['object']} (conf: {rel['confidence']:.2f})")
    print()

    # Test 3: Ask Alice about what she learned
    print("TEST 3: Recall Learned Information")
    print("-" * 70)
    response = alice.process_input("Where do I live?")
    print(f"User: Where do I live?")
    print(f"Alice: {response}")
    print()

    # Test 4: Teach more facts to build knowledge
    print("TEST 4: Building Knowledge Graph")
    print("-" * 70)
    facts = [
        "I am a software developer",
        "I created Alice",
        "Alice helps me with coding tasks"
    ]

    for fact in facts:
        response = alice.process_input(fact)
        print(f"User: {fact}")
        print(f"Alice: {response}")
        print()

    # Test 5: Check Alice's confidence and knowledge stats
    print("TEST 5: Alice's Knowledge Statistics")
    print("-" * 70)
    stats = alice.knowledge_engine.get_stats()
    print(f"Total Entities: {stats['total_entities']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Total Concepts: {stats['total_concepts']}")
    print(f"Topics Alice is Confident In: {stats['topics_confident_in']}")
    print()

    if stats['top_entities']:
        print("Most Mentioned Entities:")
        for entity, count in stats['top_entities'][:5]:
            print(f"  - {entity}: {count} mentions")
    print()

    # Test 6: Test progressive learning with same question
    print("TEST 6: Progressive Learning (Same Question 3x)")
    print("-" * 70)
    question = "who created you?"
    for i in range(3):
        print(f"\nAttempt {i+1}:")
        response = alice.process_input(question)
        print(f"User: {question}")
        print(f"Alice: {response}")

        # Check if Alice can answer independently
        can_answer, conf = alice.knowledge_engine.can_answer_independently(question, "greeting")
        print(f"Can answer independently: {can_answer} (confidence: {conf:.2f})")
    print()

    # Test 7: Check learned patterns
    print("TEST 7: Learned Response Patterns")
    print("-" * 70)
    if alice.knowledge_engine.learned_responses:
        print("Intents Alice has learned:")
        for intent, responses in list(alice.knowledge_engine.learned_responses.items())[:5]:
            print(f"  - {intent}: {len(responses)} examples")
    print()

    # Test 8: Word associations (semantic learning)
    print("TEST 8: Semantic Learning (Word Associations)")
    print("-" * 70)
    test_words = ["Alice", "Gabriel", "Waterloo", "developer"]
    for word in test_words:
        associations = alice.knowledge_engine.concepts.get_related_words(word)
        if associations:
            print(f"{word} is associated with: {', '.join([w for w, _ in associations[:5]])}")
    print()

    # Final Summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    final_stats = alice.knowledge_engine.get_stats()
    print(f"Alice learned {final_stats['total_entities']} entities")
    print(f"Alice discovered {final_stats['total_relationships']} relationships")
    print(f"Alice understands {final_stats['total_concepts']} concepts")
    print(f"Alice is confident in {final_stats['topics_confident_in']} topics")
    print()

    # Show confidence progression
    print("Topic Confidence Levels:")
    for topic, confidence in sorted(alice.knowledge_engine.topic_confidence.items(),
                                    key=lambda x: x[1], reverse=True)[:10]:
        bar = "#" * int(confidence * 20)
        print(f"  {topic:.<30} {bar} {confidence:.2f}")
    print()

    print("Test complete! Alice's knowledge has been saved to data/knowledge/")
    print()

if __name__ == "__main__":
    test_alice_knowledge()
