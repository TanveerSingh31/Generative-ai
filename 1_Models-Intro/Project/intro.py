# We will build a document similarity application
# Store 5 documents pre-loaded w/ their embeddings
# when user gives a query return the document that matches more closely with the user-query

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)



docs = [
    "MS dhoni is a cricketer",
    "Ronaldo is a soccer player",
    "Messi is a great soccer player"
]

user_query = input()


# calc. embeddings
docs_embeddings = embedding.embed_documents(docs)
userQuery_embedding = embedding.embed_query(user_query)


# Compare & find , nearest vector in docs_embeddings corresponding to user_query embedding - cosine_similarity fn.
output = cosine_similarity([userQuery_embedding], docs_embeddings)[0]

# index with closest match !
index, value = sorted(enumerate(output), key=lambda el: el[1], reverse=True)[0]


# Get element at the index with closest match corresponding to user-query
print(f"Question - {user_query}")
print("ans = ", docs[index])
print("match of value = ", value)