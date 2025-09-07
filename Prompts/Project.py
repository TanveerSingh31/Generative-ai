# CLI Bases AI application

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=api_key)


chat_history = [
    SystemMessage("You are helpful AI assistant")
]


user_query = input("User: Enter prompt - ").lower()
while(user_query != "quit"):
    # add to chat history as HumanMessage
    chat_history.append(HumanMessage(content=user_query))
    result = llm.invoke(chat_history) # sending chat_history to LLM for each query

    # add to chat history as AI Message
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")
    user_query = input("Enter prompt - ").lower()