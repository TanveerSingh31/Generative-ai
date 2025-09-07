# CLI Bases AI application

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=api_key)

user_query = input("User: Enter prompt - ").lower()
while(user_query != "quit"):
    result = llm.invoke(user_query)
    print(f"AI: {result.content}")
    user_query = input("Enter prompt").lower()