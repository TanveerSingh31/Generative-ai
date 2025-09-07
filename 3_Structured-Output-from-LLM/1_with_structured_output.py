# using Pydantic

import os
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro', google_api_key=api_key)


# Problem statement - Return =>  { name, age, email, address, gender } from person introduction 

# Step 1: Create Response class
class Person(BaseModel):
    name: str
    age: int
    email: str = Field(
        ...,
        max_length=20,
        description="Email address of the person" # (description) -> helps LLM , what data to add in here
    )
    address: str
    gender: Literal['M', 'F']




# Step 2: Make the Model Structured using -> "with_structured_output" method
# Pass schema (Class) to the fn.
structured_model = model.with_structured_output(Person)



# Step 3: Invoke the LLM to get response
result = structured_model.invoke(
"""
    Tanveer Singh this side, I have been working as Software engineer for past 2.8 years
    It is been 24 yrs. since my birth, and I have pronouns - he/him , I reside in Gurugram Haryana, and my address for email is tanveer@inc.com                             
"""
)



print(result)
