# main.py
# -*- coding: utf-8 -*-
# LangChain + llama.cpp (OpenAI-compatible) with Harmony final-channel extraction.

import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


def extract_final_with_harmony(raw_text: str) -> str:
    """Extract only the assistant's `final` channel from Harmony-formatted text.
    If no `final` message exists, avoid exposing `analysis` and return a brief notice.
    This function is robust to multiple assistant messages and stray markers.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    # Fast path: exact final marker present
    if "<|channel|>final<|message|>" in raw_text:
        # Keep last final occurrence and strip any trailing markers
        segment = raw_text.split("<|channel|>final<|message|>")[-1]
        # Split on any Harmony boundaries that may follow the final content
        segment = re.split(r"<\|end\|>|<\|start\|>|<\|call\|>|<\|return\|>", segment)[0]
        return segment.strip()

    # If Harmony tags are present but no final, do not surface analysis/commentary
    if "<|channel|>analysis<|message|>" in raw_text or "<|channel|>commentary<|message|>" in raw_text:
        return "(モデルが final チャンネルを返しませんでした。もう一度お試しください。)"

    # Otherwise return as-is (non-Harmony plain text response)
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
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2024-06\n"
            "Current date: 2025-08-13\n\n"
            "Reasoning: medium\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
            "Only return a single assistant message on the final channel. Do NOT output analysis.\n"
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [system, MessagesPlaceholder("history"), ("human", "{input}")]
    )
    # Trim long histories to stay within the model context window
    trimmer = trim_messages(
        # Keep the most recent messages
        strategy="last",
        # Use the same model for accurate token counting
        token_counter=llm,
        # Adjust this to your server/model context. Start conservative.
        max_tokens=3000,
        # Preserve system then human/tool validity
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    chain = prompt | trimmer | llm
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
        # Ensure only final channel is printed, robust to non-str content
        out = resp.content if isinstance(resp.content, str) else str(resp.content)
        print("Assistant:", extract_final_with_harmony(out))

if __name__ == "__main__":
    main()