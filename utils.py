from langchain_community.embeddings import AzureOpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

# os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
# os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

print(PINECONE_API_KEY)

embeddings = AzureOpenAIEmbeddings(
    deployment='vector-search-instance2',
    model='text-embedding-ada-002',
    openai_api_type='azure',
    azure_endpoint='https://cb-att-openai-instance.openai.azure.com/',
    api_key=AZURE_OPENAI_API_KEY,
)

pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment = 'us-east-1'
)

index = pc.Index('langchain-chatbot')
# index = 'langchain-chatbot'
print("========================", index)

def find_match(input):
    input_embed = embeddings.embed_query(input)
    result = index.query(vector=input_embed, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    client = AzureOpenAI(
        api_key = AZURE_OPENAI_API_KEY,  
        api_version = "2023-09-15-preview",
        azure_deployment="atttestgpt35turbo",
        azure_endpoint = 'https://cb-att-openai-instance.openai.azure.com/'
    )
 
    response = client.chat.completions.create(
        model="gpt-35-turbo", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpfull assistant"},
            {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that \
            would be the most relevant to provide the user with an answer from the knowledge base. \
                \n\n CONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ]
    )

    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"

    return conversation_string
