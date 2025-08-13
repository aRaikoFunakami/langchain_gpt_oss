# main.py
# -*- coding: utf-8 -*-
# LangChain + llama.cpp (OpenAI-compatible) with Harmony final-channel extraction.

import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


def extract_final_with_harmony(raw_text: str) -> str:
    """Parse Harmony tokens and return only the assistant final message.
    Falls back to regex if tokens are already combined in a single string.
    """
    # Fast path: if tags are present, take final channel content.
    if "<|channel|>final<|message|>" in raw_text:
        # Keep last final message chunk; strip trailing end marker if present.
        part = raw_text.split("<|channel|>final<|message|>")[-1]
        # Remove potential closing markers.
        part = re.split(r"<\|end\|>|<\|start\|>", part)[0].strip()
        return part

    # If server gave us already-clean text, just return it.
    return raw_text

def build_chain(base_url: str, model: str, api_key: str = "dummy-key"):
    llm = ChatOpenAI(
        base_url=base_url,   # e.g. "http://localhost:8080/v1"
        api_key=api_key,     # llama.cpp usually ignores auth; any string ok
        model=model,         # e.g. "gpt-oss-20b"
        temperature=0.7,
        verbose=True,
    )

    system = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "Respond in Japanese unless asked otherwise.\n"
            "Do not reveal chain-of-thought; provide only final answers."
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [system, MessagesPlaceholder("history"), ("human", "{input}")]
    )
    chain = prompt | llm
    return chain

def main():
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    model = os.getenv("MODEL_NAME", "gpt-oss-20b")

    chain = build_chain(base_url, model)
    history = InMemoryChatMessageHistory()
    with_history = RunnableWithMessageHistory(
        chain,
        lambda _: history,
        input_messages_key="input",
        history_messages_key="history",
    )

    print("Chat started. Type 'exit' to quit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        resp = with_history.invoke({"input": user}, config={"configurable": {"session_id": "cli"}})
        # Ensure only final channel is printed
        print("Assistant:", extract_final_with_harmony(resp.content))

if __name__ == "__main__":
    main()