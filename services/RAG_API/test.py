import os
from dotenv import load_dotenv

print("=== Testing Configuration ===")

# Load .env
load_dotenv("../../.env")

print(f"AZURE_SEARCH_ENDPOINT: {os.getenv('AZURE_SEARCH_ENDPOINT')}")
print(f"AZURE_SEARCH_API_KEY: {os.getenv('AZURE_SEARCH_API_KEY')[:20]}...")
print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_SEARCH_INDEX: {os.getenv('AZURE_SEARCH_INDEX')}")

if all([
    os.getenv('AZURE_SEARCH_ENDPOINT'),
    os.getenv('AZURE_SEARCH_API_KEY'),
    os.getenv('AZURE_OPENAI_ENDPOINT')
]):
    print("\n All required variables loaded!")
else:
    print("\n Missing variables!")