# This file contains the streamlit application for a GenAI Marketing Assisant
# Please remember to fill in all the required elements

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent*
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import os


# ------ Load your dataset ---------
df = pd.read_csv(r'path to your data') # Inset the link to your own data

# ------- Initialize the Chat LLM ---------
os.environ['OPENAI_API_KEY']='Your Own Key' # Insert your own key from OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the Pandas Agent from experimental module
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True) # Be careful with confidential data. In my case I allowed the LLM to run code arbitrarely since I don't have any sensitive data.

st.title("GenAI Marketing Assistant")

user_input = st.text_input("Ask a question about the marketing data:")

if user_input:
    with st.spinner("Analyzing..."):
        response = agent.run(user_input)
    st.write(response)
