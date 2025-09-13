import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv


# Problem statement - A fintech company wants to build an AI-powered FAQ assistant for their payments app. The assistant should take a customer’s query and process it in multiple sequential steps using LangChain’s

class Query(BaseModel):
    query_type: Literal['payments', 'orders', 'service']

class Query2(BaseModel):
    responses: list[str]



load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)

structured_llm = llm.with_structured_output(Query)
structured_llm2 = llm.with_structured_output(Query2)


prompt1 = PromptTemplate(
    template="Classify the user query {user_query} into 'payments', 'orders', 'service' ",
    input_variables=['user_query']
)

def log_fn(x):
    print("\n[LOG] Passing through:", x.query_type)
    return x.query_type   # must return the same value to keep chain intact


prompt2 = PromptTemplate(
    template="respond to the user that the issue is linked to - {query}",
    input_variables=['query']   
)




chain = prompt1 | structured_llm | log_fn | prompt2 | structured_llm2


result = chain.invoke({'user_query': "My payment is stuck for past 2 days, how can i resolve this issue ?"})


print(result.responses[0])