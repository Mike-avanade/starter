"""
End-to-end smoke test for Document Assistant.
Tests QA, summarization, and calculation intents.
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.assistant import DocumentAssistant

load_dotenv()

TEST_QUERIES = [
    {
        "query": "What is the total amount in invoice INV-001?",
        "expected_intent": "qa",
        "description": "Q&A about a specific document amount"
    },
    {
        "query": "Summarize all contracts in the system.",
        "expected_intent": "summarization",
        "description": "Summarization task"
    },
    {
        "query": "Calculate the total of all invoices over $50,000.",
        "expected_intent": "calculation",
        "description": "Calculation across multiple documents"
    }
]

def run_smoke_test():
    """Run smoke test across all three intent types."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set")
        return False

    print("=" * 60)
    print("SMOKE TEST: Document Assistant End-to-End")
    print("=" * 60)

    assistant = DocumentAssistant(
        openai_api_key=api_key,
        model_name="gpt-4o",
        temperature=0.1
    )

    user_id = "test_user"
    session_id = assistant.start_session(user_id)
    print(f"✅ Session created: {session_id}\n")

    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"Test {i}: {test['description']}")
        print(f"Query: {test['query']}")
        
        try:
            result = assistant.process_message(test["query"])
            
            if not result.get("success"):
                print(f"❌ FAIL: {result.get('error')}\n")
                results.append(False)
                continue

            intent = result.get("intent", {})
            intent_type = intent.get("intent_type")
            confidence = intent.get("confidence", 0)

            # Check intent type
            if intent_type == test["expected_intent"]:
                print(f"✅ Intent: {intent_type} (confidence: {confidence:.2f})")
            else:
                print(f"⚠️  Intent: {intent_type} (expected: {test['expected_intent']}, confidence: {confidence:.2f})")

            # Check response
            response = result.get("response")
            if response:
                print(f"✅ Response: {response[:100]}...")
            else:
                print(f"❌ No response generated")
                results.append(False)
                continue

            # Check sources
            sources = result.get("sources") or result.get("active_documents") or []
            tools_used = result.get("tools_used", [])
            
            print(f"🔍 Sources: {sources if sources else 'None'}")
            print(f"🛠️  Tools used: {tools_used if tools_used else 'None'}")
            
            results.append(True)

        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}\n")
            results.append(False)
        
        print()

    # Verify logs and session files were created
    print("=" * 60)
    print("ARTIFACT VERIFICATION")
    print("=" * 60)
    
    logs_exist = os.path.exists("./logs") and len(os.listdir("./logs")) > 0
    session_file = os.path.join("./sessions", f"{session_id}.json")
    session_exists = os.path.exists(session_file)

    print(f"{'✅' if logs_exist else '❌'} Tool logs created: ./logs/")
    print(f"{'✅' if session_exists else '❌'} Session file created: {session_file}")

    if session_exists:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            print(f"   - Session has {len(session_data.get('conversation_history', []))} conversation turns")

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)