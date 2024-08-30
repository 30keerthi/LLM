import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.llms import CTransformers
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from streamlit_chat import message
from langchain_community.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain


# Function to load CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to compute statistics
def compute_statistics(df):
    statistics = {
        'mean': df.mean(numeric_only=True),
        'median': df.median(numeric_only=True),
        'mode': df.mode().iloc[0],
        'std_dev': df.std(numeric_only=True),
        'correlation': df.corr(numeric_only=True)
    }
    return statistics

# Function to generate plots
def plot_histogram(df, column):
    plt.hist(df[column])
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)
    plt.clf()

def plot_scatter(df, column1, column2):
    plt.scatter(df[column1], df[column2])
    plt.title(f'Scatter Plot between {column1} and {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    st.pyplot(plt)
    plt.clf()

def plot_line(df, column):
    plt.plot(df[column])
    plt.title(f'Line Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    st.pyplot(plt)
    plt.clf()

# Function to load the LLM model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_file = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


# Streamlit app layout
st.title(" Data Exploration ")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:

   # Using  temporary file as the CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()

   # Embedding the data using HuggingFace Models

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

   # FAISS to build an index for efficient similarity search.
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function to create a converssational chat
    def conversational_chat(query):
       result = chain({"question": query, "chat_history": st.session_state['history']})
       st.session_state['history'].append((query, result["answer"]))
       return result["answer"]


    if 'history' not in st.session_state:
       st.session_state['history'] = []

    if 'generated' not in st.session_state:
       st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name]

    if 'past' not in st.session_state:
       st.session_state['past'] = ["Hey"]

   # Create ontainer for the chat history
    response_container = st.container()

   # Create container for the user's input
    container = st.container()

    with container:
       with st.form(key='my_form', clear_on_submit=True):
           user_input = st.text_input("Query:", placeholder="Query your data", key='input')
           submit_button = st.form_submit_button(label='Send')

       if submit_button and user_input:
           output = conversational_chat(user_input)

           st.session_state['past'].append(user_input)
           st.session_state['generated'].append(output)

    if st.session_state['generated']:
       with response_container:
           for i in range(len(st.session_state['generated'])):
               message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="bottts")
               message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    st.write("Statistics:")
    stats = compute_statistics(df)
    st.write(stats)

    columns = df.columns.tolist()
    selected_plot = st.selectbox("Choose a plot type", ["Histogram", "Scatter Plot", "Line Plot"])

    if selected_plot == "Histogram":
        column = st.selectbox("Choose a column for the histogram", columns)
        if st.button("Generate Histogram"):
            plot_histogram(df, column)

    elif selected_plot == "Scatter Plot":
        column1 = st.selectbox("Choose the first column for scatter plot", columns)
        column2 = st.selectbox("Choose the second column for scatter plot", columns)
        if st.button("Generate Scatter Plot"):
            plot_scatter(df, column1, column2)

    elif selected_plot == "Line Plot":
        column = st.selectbox("Choose a column for the line plot", columns)
        if st.button("Generate Line Plot"):
            plot_line(df, column)
