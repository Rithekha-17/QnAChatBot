import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ['LANGCHAIN_PROJECT']="Q&A ChatBot with OLLAMA"

#prompt template

prompt=ChatPromptTemplate([
    ("system","you are a helpful assistant. please response to the user queries"),
    ("user","Question: {question}")
])

def generate_response(question, engine, temperature, max_tokens):
    llm=Ollama(model=engine)
    parser=StrOutputParser()
    chain=prompt|llm|parser
    answer=chain.invoke({'question':question})
    return answer

## app
st.title("Q&A ChatBot using Ollama")
st.sidebar.title("Settings")
llm=st.sidebar.selectbox("Select an Ollama Model: ",["gemma:2b"])
temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens=st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any questions")
user_input=st.text_input("You: ")

if user_input:
    response=generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")