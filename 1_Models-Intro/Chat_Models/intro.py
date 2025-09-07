from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# temperature controls the creativity in the response from LLM
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0, max_output_tokens=1000)

result = chatModel.invoke('write a joke')

print(result)