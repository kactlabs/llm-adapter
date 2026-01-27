#!/usr/bin/env python3
"""
Simple CLI Chatbot using LLM Factory with LangChain
"""

import warnings
# Suppress FutureWarning from google.generativeai (internal to langchain-google-genai)
warnings.filterwarnings('ignore', category=FutureWarning, module='langchain_google_genai')

from llm import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def main():
    """Run the CLI chatbot"""
    print("=" * 50)
    print("Welcome to the CLI Chatbot!")
    print("=" * 50)
    print("\nType 'quit', 'exit', or 'q' to end the conversation\n")
    
    # Initialize LLM
    llm = get_llm()
    
    # Setup message history
    chat_history = InMemoryChatMessageHistory()
    
    # Create prompt template for one-line answers
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that provides concise one-line answers only."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain with message history
    chain = prompt | llm
    
    conversation = RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        try:
            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}}
            )
            print(f"Bot: {response.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
