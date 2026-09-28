"""
test_brief.py — DealMind: Test generate_meeting_brief()
========================================================
Run with:  py test_brief.py
"""

import json
from llm_service import generate_meeting_brief

# Mock memories — these represent what Hindsight would return for Rahul.
MEMORIES = [
    "Rahul from ABC Technologies considers Rs. 50,000 too expensive.",
    "Rahul is evaluating Salesforce.",
    "Rahul needs advanced analytics.",
    "Rahul needs automated reporting.",
    "The sales representative promised revised enterprise pricing.",
]

if __name__ == "__main__":
    print("=" * 60)
    print("DealMind — generate_meeting_brief() Test")
    print("=" * 60)
    print("\nInput memories (from Hindsight):\n")
    for i, memory in enumerate(MEMORIES, 1):
        print(f"  {i}. {memory}")
    print("\n" + "-" * 60)

    brief = generate_meeting_brief(MEMORIES)

    print("\nMeeting Brief Result:\n")
    print(json.dumps(brief, indent=2))
    print("\n" + "=" * 60)
    print("Test complete.")
