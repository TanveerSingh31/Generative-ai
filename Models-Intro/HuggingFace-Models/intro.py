import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv('HUFFINGFACEHUB_ACCESS_TOKEN')


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    temperature=2.0
)


llm = ChatHuggingFace(llm=llm)

result = llm.invoke("Tell me a joke")

print(result)


