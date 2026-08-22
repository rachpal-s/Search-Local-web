from dotenv import load_dotenv
import os, urllib.request

load_dotenv()
key = os.getenv("OLLAMA_API_KEY")
print("Key loaded:", repr(key))

if key:
    req = urllib.request.Request(
        "https://ollama.com/api/whoami",
        headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print("Status:", resp.status)
            print(resp.read().decode())
    except urllib.error.HTTPError as e:
        print("Status:", e.code)
        print(e.read().decode())
else:
    print("Key is None even after load_dotenv() — check .env file content directly:")
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            if "OLLAMA_API_KEY" in line:
                print(repr(line))