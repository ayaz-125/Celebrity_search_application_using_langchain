# In this application we use the perplecity api and prompt template also 

# Integrate our code Perp API
import os
from constant import pplx_api_key 
from langchain_community.chat_models import ChatPerplexity
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory


import streamlit as st

os.environ["PERPLEXITY_API_KEY"] = pplx_api_key

# Streamlit framework

st.title("Celebrity Search Result")
input_text = st.text_input("Search the topic u want")



# chatperplexity llm
llm = ChatPerplexity(
    model="sonar",  
    temperature=0.7,
    pplx_api_key=pplx_api_key  
)


# Prompt Template - 1

first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about the celebrity {name}"

)

# Memory 
person_memory = ConversationBufferMemory(input_key = "name", memory_key = "chat_history")
dob_memory = ConversationBufferMemory(input_key = "person", memory_key = "chat_history")
descr_memory = ConversationBufferMemory(input_key = "dob", memory_key = "description_history")

llm_chain1 = LLMChain(llm=llm,prompt = first_input_prompt,verbose =True,output_key = 'person',memory = person_memory)


# prompt Template - 2
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born"
)

llm_chain2 = LLMChain(llm=llm,prompt = second_input_prompt,verbose =True,output_key = 'dob',memory = dob_memory)

# prompt Template - 3
third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major event around {dob} in the world"
)

llm_chain3 = LLMChain(llm=llm,prompt = third_input_prompt,verbose =True,output_key = 'description',memory= descr_memory)



# Main chain
parent_chain = SequentialChain(chains = [llm_chain1,llm_chain2,llm_chain3],input_variables = ['name'],output_variables = ['person','dob',"description"],verbose = True)


if input_text:
    response = parent_chain.invoke({"name": input_text})
    st.write(response)  
    # st.write(response['dob'])    # also fetch in the form of key pair value 

    with st.expander("Person Name"):
        st.info(person_memory.buffer)


    with st.expander("Major Events"):
        st.info(descr_memory.buffer)













# In simplesequential chain i was getting the last information
# with the help of sequential chain i will get the entire information in the json form
    


