import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from text import text
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
hf_access_token = os.getenv('HUFFINGFACEHUB_ACCESS_TOKEN')


# Problem Statement
# Create Notes & Quiz parallely for a piece of text - using parallel chains
# Merge the generate response into 1 single doc -> using sequential chain


# Model 1 : to be used to make notes from text
GeminiModel = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)


# Model 2: will be used to create a 5 ques. quiz from text
LlamaModel = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="text-generation",
    huggingfacehub_api_token=hf_access_token
)


prompt1 = PromptTemplate(
    template="Create short and simple notes for the text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Create a short 5 question quiz for the following text \n {text}",
    input_variables=['text']
)


parser = StrOutputParser()



# create parallel chains
chain1 = RunnableParallel(
    notes =  prompt1 | GeminiModel | parser ,
    quiz = prompt2 | LlamaModel | parser
)


prompt3 = PromptTemplate(
    template="Create a single document, containing notes - {notes} , and quiz - {quiz}",
    input_variables=['notes', 'quiz']
)


# Sequential chain
chain2 = prompt3 | GeminiModel | parser



# Merging Parallel chain with sequential chain
chain3 = chain1 | chain2


result = chain3.invoke({'text': text})

print(result)
