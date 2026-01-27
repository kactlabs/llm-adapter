#!/usr/bin/env python3
"""
Simple CLI Chatbot using LLM Factory with LangChain
"""

from llm import get_llm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


def main():
    """Run the CLI chatbot"""
    print("=" * 50)
    print("Welcome to the CLI Chatbot!")
    print("=" * 50)
    print("\nType 'quit', 'exit', or 'q' to end the conversation\n")
    
    # Initialize LLM
    llm = get_llm()
    
    # Setup LangChain memory
    memory = ConversationBufferMemory()
    
    # Create prompt template for one-line answers
    template = """The following is a conversation between a human and an AI assistant. 
The AI provides concise one-line answers only.

Current conversation:
{history}
Human: {input}
AI:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
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
            response = conversation.predict(input=user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
