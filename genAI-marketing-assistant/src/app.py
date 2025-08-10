import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import os

#os.environ['SSL_CERT_FILE'] = '/path/to/anaconda3/ssl/certs/ca-certificates.crt'


# Load your dataset
df = pd.read_csv(r'C:\Users\Fatima\OneDrive\Desktop\genAI-marketing-assistant\genAI-marketing-assistant\Data\data_cleaned.csv')

# Initialize the Chat LLM
os.environ['OPENAI_API_KEY']='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'
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the Pandas Agent from experimental module
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True) # Be careful with confidential data. In my case I allowed the LLM to run code arbitrarely since I don't have any sensitive data.

st.title("GenAI Marketing Assistant")

user_input = st.text_input("Ask a question about the marketing data:")

if user_input:
    with st.spinner("Analyzing..."):
        response = agent.run(user_input)
    st.write(response)
