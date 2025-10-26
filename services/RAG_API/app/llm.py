import os
from dotenv import load_dotenv; load_dotenv()
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZ_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
)
EMBED = os.getenv("AZ_OPENAI_EMBED_DEPLOY")
CHAT  = os.getenv("AZ_OPENAI_CHAT_DEPLOY")

def embed_texts(texts):  # returns list[list[float]]
    r = client.embeddings.create(model=EMBED, input=texts)
    return [x.embedding for x in r.data]

def chat(messages, temperature=0.1):
    r = client.chat.completions.create(model=CHAT, messages=messages, temperature=temperature)
    return r.choices[0].message
