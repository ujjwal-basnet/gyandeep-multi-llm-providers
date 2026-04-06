import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from sarvamai import SarvamAI

async def test():
    api_key = os.getenv("SARVAMAI_KEY")
    if not api_key or api_key == "your_key_here":
        print("API Key not found or empty.")
        return

    client = SarvamAI(api_subscription_key=api_key)
    try:
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-2B-chat", # try a different model just in case? wait, let's try the current one first
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print("Success:", response)
    except Exception as e:
        print("Error with sarvam-2B-chat:", e)
        
    try:
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-30b",
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print("Success:", response)
    except Exception as e:
        print("Error with sarvam-30b:", e)

if __name__ == "__main__":
    asyncio.run(test())
