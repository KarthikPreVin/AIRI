from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os 

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key==None:
    print("API NO FOUND")
    exit()
os.environ["GOOGLE_API_KEY"] = key



emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "i am Rise of Devil",
    "i like chocolate milkshake",
    "u serve me for eternity"
]


db = FAISS.from_texts(texts,emb_model)
query = "Who am i?"
results = db.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")



llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", temperature=0.7)

# Basic prompt
response = llm.invoke("who are you?")
print(response)

##add vector embedings for memory
