from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if key==None:
    print("API NO FOUND")
    exit()
os.environ["GOOGLE_API_KEY"] = key

model = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", temperature=0.7)

user_template = f"""
you are an emphathetic AI chatbot.
**STRICTLY KEEP RESPONSE WITHIN 100 WORDS OR 1 PARAGRAPH**. 
Do not harass the user.
Do not threaten. 
always chat with a calm and caring tone. 
respond to the message now :
{{message}}
"""

conversation_history = []
parser = StrOutputParser()


def chat(message):
    conversation_history.append(("human",message))
    prompt = ChatPromptTemplate.from_messages(
        conversation_history
    )
    chain = prompt | model | parser

    result = chain.invoke({"message": message})
    print(f"YOU:\n\t{message}\nAIRI:\n\t{result}")
    conversation_history.append(("ai",result))
    return result

chat("help")
chat("im bob")
chat("what is my name?")