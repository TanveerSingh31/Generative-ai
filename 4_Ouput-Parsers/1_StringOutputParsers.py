import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')


model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=api_key)

# using Output Parsers -> string output parser
# We donot need to extract the content from the response of the LLM


# Usecase -> 
# Step 1 : get detailed report from LLM on a topic
# Step 2 : send the detailed report to LLM and ask to summarize in 5 points

# template 1
template = PromptTemplate(
    template="Give a detailed report on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

# template 2
template2 = PromptTemplate(
    template="Give 5 pointers summary of the following report - {text}",
    input_variables=['text']
)


# Chain -> 
# Use template1 to ask input from the user
# pass the prompt to model
# parse the output from model -> strOutputParser
# pass the output to prompt2
# pass the new prompt to model
# parse the output from model

chain = template | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'MS Dhoni'})

print(result)
