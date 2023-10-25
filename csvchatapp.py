import streamlit as st 
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import SmartDataframe, Agent
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def chat_with_csv_mock(df, prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner('Processing...'):
        time.sleep(1)
        result = "AI answer to your query"
        # Get Clarification Questions
        questions = []
        if st.session_state.followup:
            questions = ["What else can I do?", "Does this make sense", "Where are we?"]
        # Explain how the chat response is generated
        explanation = None
        if st.session_state.explain:
            explanation = "This is why I came up with this answer blah blah blah..."
        st.session_state.messages.append({"role": "assistant", "content": result, "explanation": explanation})
    return result, questions, explanation

def chat_with_csv(df, prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner('Processing...'):
        result = df.chat(query=prompt, output_type="string")
        # Get Clarification Questions
        questions = []
        if st.session_state.followup:
            questions = df.clarification_questions(prompt)
        # Explain how the chat response is generated
        explanation = None
        if st.session_state.explain:
            explanation = df.explain()
            explanation = f"{explanation}\n```\n{df._lake.last_code_executed}\n```"
        st.session_state.messages.append({"role": "assistant", "content": result, "explanation": explanation})
    return result, questions, explanation

@st.cache_resource(max_entries=1)
def get_dfagent(dfs, model):
    print(f'Using model {model}')
    llm = OpenAI(api_token=openai_api_key, model=model)
    df = Agent(dfs, config={"llm": llm, "verbose": True}, memory_size=10)
    return df

def reset_chat():
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.cache_resource.clear()

st.set_page_config(layout='wide')
st.title("MMM AI Demo App")

with st.sidebar:
    st.header('Settings')
    st.session_state.model = st.selectbox(
        'Model',
        ('gpt-3.5-turbo', 'gpt-3.5-turbo-instruct', 'gpt-4'),
        label_visibility='collapsed', key='model_selector'
        )
    st.button("Reset Chat", type="primary", on_click=reset_chat)
    st.session_state.mock = st.toggle("Mock mode")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'], accept_multiple_files=True)

if input_csv:
    dfs = []
    for file in input_csv:
        data = pd.read_csv(file)
        dfs.append(data)
        with st.expander(f"See {file.name} dataframe"):
            st.dataframe(data, use_container_width=False)
    # df = SmartDataframe(data, config={"llm": llm, "verbose": True})
    # df = Agent(dfs, config={"llm": llm, "verbose": True}, memory_size=10)
    df = get_dfagent(dfs, st.session_state.model)
    
    scol1, scol2 = st.columns([1,1])
    with scol1:
        st.toggle('Enable follow-ups', key='followup')
    with scol2:
        st.toggle('Enable explanation', key='explain')
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("explanation"):
                with st.expander("See explanation"):
                    st.markdown(message["explanation"])

    if prompt := st.chat_input("What do you want to do?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if st.session_state.mock:
                result, questions, explanation = chat_with_csv_mock(df, prompt)
            else:
                result, questions, explanation = chat_with_csv(df, prompt)
            message_placeholder.markdown(result)
            if explanation:
                with st.expander("See explanation"):
                    st.markdown(explanation)
        for q in questions:
            st.button(q, on_click=chat_with_csv, args=[df, q])