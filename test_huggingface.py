import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError(
        "HF_TOKEN is missing. Add your Hugging Face token to the .env file."
    )

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=token,
)

print("Connecting to Hugging Face...")

response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a concise assistant.",
    input="Reply with exactly: DealMind AI is working!",
    reasoning={
        "effort": "low"
    },
)

print("\nModel response:")
print(response.output_text)