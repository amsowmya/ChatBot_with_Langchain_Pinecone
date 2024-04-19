import os
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st 
from streamlit_chat import message
from utils import *

load_dotenv()

# os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

st.title("Smart Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I assist you']

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_deployment="atttestgpt35turbo",
    azure_endpoint="https://cb-att-openai-instance.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY
)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question
as truthfully as possing using the provided context.
And if the answer is not contained within the context below. Say, I don't know""")

human_msg_template = HumanMessagePromptTemplate.from_template(template='{input}')

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, 
                                                    MessagesPlaceholder(variable_name='history'),
                                                    human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory,
                                 llm = llm,
                                 prompt=prompt_template,
                                 verbose=True)

# container for chat history
response_container = st.container()

# container for text box
text_container = st.container()

with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            response = conversation.run(input=f"Context:\n {context} \n\nQuery: {query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i)+"_user")