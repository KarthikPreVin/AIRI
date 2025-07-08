from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os 
import faiss

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key==None:
    print("API NO FOUND")
    exit()
os.environ["GOOGLE_API_KEY"] = key

texts = [
    "You are gemma3n",
    "i call you jarvis",
    "i am Rise of Devil",
    "i like chocolate milkshake",
    "u serve me for eternity"
]

memory_prompt = """
REPLY USING CONTEXT MEMORY BELOW:
{memory}
PROMPT:
{prompt}
"""

llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", temperature=0.7)
emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(["memory"],emb_model)
# d = emb_model.client.get_sentence_embedding_dimension()
# index = faiss.IndexFlatL2(d)

def add_text(text):
    db.add_texts([text])

def search(question,k=2):
    return [i.page_content for i in db.similarity_search(question,k=k)]

def ask(question):
    res = search(question)
    prepared_prompt = memory_prompt.format(memory = "\n".join(res),prompt=question)
    response = llm.invoke(prepared_prompt)
    db.add_texts([question])
    return response



for i in texts:
    add_text(i)

a1 = ask("who are you?")
a2 = ask("who am i?")

print(a1)
print(a2)



# Basic prompt
# response = llm.invoke("who are you?")
# print(response)

##add vector embedings for memory
