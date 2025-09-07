import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

print(api_key)

llm = GoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=api_key)

result = llm.invoke('when were you last trained')

print(result)
