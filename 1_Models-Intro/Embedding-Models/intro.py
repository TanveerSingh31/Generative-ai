from langchain_huggingface import HuggingFaceEmbeddings

llm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device': 'cpu'})

result = llm.embed_query("What is capital of India ?")

# Will return vector-embedding from the user query !
print(result)
