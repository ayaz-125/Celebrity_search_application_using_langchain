# This is just the basic application using streamlit and perplexity api without any prompt and all 

# Integrate our code Perp API
import os
from constant import pplx_api_key 
from langchain_community.chat_models import ChatPerplexity

import streamlit as st

os.environ["PERPLEXITY_API_KEY"] = pplx_api_key

# Streamlit framework

st.title("Langchain Demo with PERPLEXITY API")
input_text = st.text_input("Search the topic u want")

# chatperplexity llm
llm = ChatPerplexity(
    model="sonar",  
    temperature=0.7,
    pplx_api_key=pplx_api_key  
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)