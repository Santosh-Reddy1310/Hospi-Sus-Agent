"""
Utility to save provider API keys to a local .env file (ignored by git).
Run this locally — do NOT paste secrets into chat.

Usage (Windows):
  .venv\Scripts\python scripts\save_key.py

This will prompt for a provider and a key and write a `.env` file in the repo root.
The `.env` file is included in `.gitignore` so it won't be committed.
"""
import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / '.env'

def main():
    print("This utility will save your LLM provider key to a local .env file (ignored by git).")
    print("Do NOT share this key or paste it into public chat/PRs.")
    provider = input("Provider (openai/groq) [groq]: ").strip() or 'groq'
    # Use getpass so the key doesn't show on-screen
    key = getpass.getpass(f"Enter {provider} API key: ")
    if not key:
        print("No key entered — aborting.")
        return

    lines = [
        f"ENABLE_LLM=1",
        f"LLM_PROVIDER={provider}",
    ]
    if provider.lower() == 'groq':
        lines.append(f"GROQ_API_KEY={key}")
    elif provider.lower() == 'openai':
        lines.append(f"OPENAI_API_KEY={key}")
    else:
        print("Unknown provider — saving generic key variable.")
        lines.append(f"{provider.upper()}_API_KEY={key}")

    # Write .env
    with open(ENV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Saved .env to {ENV_PATH}")
    print("Remember: do NOT commit .env to source control. Rotate the key if it was shared publicly.")

if __name__ == '__main__':
    main()
