import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_insights(df):
    class LineSeparatedListOutputParser(BaseOutputParser):
        """Parse the output of an LLM call to a line-separated list."""
        def parse(self, text: str):
            """Parse the output of an LLM call."""
            return text.strip().split("\n")

    template = """You are an expert marketing mix modelling assistant who generates a lists of insights.
    A user will pass in modelled data in the form of a CSV table, and you should generate 3 insights based on that data
    in the form of a line separated (`\n`) list. ONLY return a line separated list, and nothing more. Your insights should
    be concise and only be based on the data provided"""
    human_template = "{data}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chat_model = ChatOpenAI(
        model='gpt-3.5-turbo',
        verbose=True,
        temperature=0.1,
        openai_api_key=openai_api_key,
        max_tokens=256
    )
    chain = chat_prompt | chat_model | LineSeparatedListOutputParser()
    result = chain.invoke({"data": f"{df.to_csv(index=False).strip()}"})
    result = [x for x in result if len(x)]
    return result

def main():
    st.title("MMM AI Demo App")

    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        st.sidebar.header("Visualizations")
        plot_options = ["Line plot", "Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        data = pd.read_csv(data_file)
        with st.expander(f"Dataframe"):
            st.dataframe(data, use_container_width=False)
        genbutton = st.button("Generate Insights", type="primary")
        col1, col2 = st.columns(2)
        with col1:
            if selected_plot == "Line plot":
                x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
                y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
                st.write("Line plot:")
                fig, ax = plt.subplots()
                sns.lineplot(x=data[x_axis], y=data[y_axis], ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)

            if selected_plot == "Bar plot":
                x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
                y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
                st.write("Bar plot:")
                fig, ax = plt.subplots()
                sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
                # bins = st.sidebar.slider("Number of bins", 5, 100, 10)
                # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=bins))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig)

            elif selected_plot == "Scatter plot":
                x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
                y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
                st.write("Scatter plot:")
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
                st.pyplot(fig)

            elif selected_plot == "Histogram":
                column = st.sidebar.selectbox("Select a column", data.columns)
                bins = st.sidebar.slider("Number of bins", 5, 100, 20)
                st.write("Histogram:")
                fig, ax = plt.subplots()
                sns.histplot(data[column], bins=bins, ax=ax)
                st.pyplot(fig)

            elif selected_plot == "Box plot":
                column = st.sidebar.selectbox("Select a column", data.columns)
                st.write("Box plot:")
                fig, ax = plt.subplots()
                sns.boxplot(data[column], ax=ax)
                st.pyplot(fig)
        with col2:
            if genbutton:
                insights = generate_insights(data)
                for i in insights:
                    st.info(i)

if __name__ == "__main__":
    main()