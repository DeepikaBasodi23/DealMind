"""
test_analysis.py — DealMind: Test analyze_conversation()
=========================================================
Run with:  py test_analysis.py
"""

import json
from llm_service import analyze_conversation

CONVERSATION = (
    "Rahul from ABC Technologies likes our product but thinks Rs. 50,000 is expensive. "
    "He is also evaluating Salesforce. "
    "His team needs advanced analytics and automated reporting. "
    "I promised to send revised enterprise pricing tomorrow."
)

if __name__ == "__main__":
    print("=" * 60)
    print("DealMind — analyze_conversation() Test")
    print("=" * 60)
    print("\nInput conversation:\n")
    print(f"  {CONVERSATION}\n")
    print("-" * 60)

    result = analyze_conversation(CONVERSATION)

    print("\nStructured Analysis Result:\n")
    print(json.dumps(result, indent=2))
    print("\n" + "=" * 60)
    print("Test complete.")
