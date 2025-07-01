from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key==None:
    print("API NO FOUND")
    exit()
os.environ["GOOGLE_API_KEY"] = key


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Basic prompt
response = llm.invoke("Explain black holes in simple terms.")

print(response)
response = llm.invoke("what did i ask you?")
print(response)