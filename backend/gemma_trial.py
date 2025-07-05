

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key==None:
    print("API NO FOUND")
    exit()
os.environ["GOOGLE_API_KEY"] = key

llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", temperature=0.7)

# Basic prompt
response = llm.invoke("who are you?")
print(response)

##add vector embedings for memory
