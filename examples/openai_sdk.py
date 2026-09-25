"""Use the gateway from the official OpenAI SDK: only base_url and api_key change.

GATEWAY_KEY=gk_... uv run python examples/openai_sdk.py
"""

import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:58080/v1", api_key=os.environ["GATEWAY_KEY"])

resp = client.chat.completions.create(
    model="chat-default",  # an alias; the gateway decides which provider serves it
    messages=[{"role": "user", "content": "Explain circuit breakers in one sentence."}],
    temperature=0,  # temperature 0 → eligible for the exact cache
)
print(resp.choices[0].message.content)

# Streaming works the same way
for chunk in client.chat.completions.create(
    model="chat-default", messages=[{"role": "user", "content": "Count to five."}], stream=True
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# Gateway headers (provider actually used, attempts, cache) are on the raw response
raw = client.chat.completions.with_raw_response.create(
    model="chat-default", messages=[{"role": "user", "content": "hi"}]
)
print({k: v for k, v in raw.headers.items() if k.startswith("x-gateway")})
