import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


name = input("Research Paper name - ")
output_length = input("output length ? ")
type = input("type of paper - wordy or generic ? ")


# Step1: Create template
template = ChatPromptTemplate.from_template(
"""
    You are a Research paper analyst.
    I want to a summarized version of the follwing research paper.
    name - {name}
    length - {output_length}
    type - {type}
    If paper is not found, return an error - "Paper not found"
"""
)


# Step 2: Add values to Template -> generate prompt string
prompt = template.invoke({
    "name": name,
    "output_length": output_length,
    "type": type
})



# # Step 3: Give Prompt to LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
result = llm.invoke(prompt)


print(result.content)




