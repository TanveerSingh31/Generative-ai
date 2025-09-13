import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from text import text
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
GeminiModel = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)


class FeedbackModel(BaseModel): 
    sentiment: Literal['positive', 'negative']

GeminiModelStrucutured = GeminiModel.with_structured_output(FeedbackModel)


prompt1 = PromptTemplate(
    template="Respond with a positive response to the user feedback - {feedback}",
    input_variables=['feedback']
)

prompt2 = PromptTemplate(
    template="Respond with a Negative response to the user feedback - {feedback}",
    input_variables=['feedback']
)


# Problem Statement - generate a response based on the user feedback , positive / Negative
# Step 1 : Classify the feedback into positive / negative
# Step 2 : run chain1 if positive , else run chain2 if negative

prompt = PromptTemplate(
    template="Classify the user feedback into positive or negative - {feedback}",
    input_variables=['feedback']
)

parser = StrOutputParser()


chain = prompt | GeminiModelStrucutured

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt1 | GeminiModel | parser),
    (lambda x:x.sentiment == 'negative', prompt2 | GeminiModel | parser),
    lambda x: "Your feedback was not positive , nor negative",
)


chain_final = chain | branch_chain

result = chain_final.invoke({'feedback': "This is a very good product"})


print(result)


