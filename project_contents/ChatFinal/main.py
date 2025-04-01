import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import PyPDF2
from io import BytesIO
import os
import tempfile






import chardet


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Genrative AI Chatbot using Zephyr (small LLM)"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi, I can answer all your queries regarding uploaded document"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
        streaming=True,
        model_path="/Users/devgup/Documents/Documents/MBAI/KiranSimpleRAG/zephyr-7b-beta.Q4_K_S.gguf",
        temperature=0.8,
        top_p=0.95,
        min_p=0.10,
        verbose=True,
        n_ctx=1024,
        n_batch=126
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain





def home():
    st.title('Home Page')
    # Add content for home page

def trend_analysis():
    st.title('Trend Analysis')
    dc = os.path.join(os.path.dirname(__file__), '../plots/death_count.png')
    d = os.path.join(os.path.dirname(__file__), '../plots/demographics.png')
    pct = os.path.join(os.path.dirname(__file__), '../plots/patient_condition_trend.png')
    s = os.path.join(os.path.dirname(__file__), '../plots/symptoms.png')
    st.image(dc, caption='Death Count of High Risk Patients')
    st.image(d, caption='Demographics of High Risk Patients')
    st.image(pct, caption='Condition Trend in High Risk Patients (Normalized)')
    st.image(s, caption='Top 10 Symptoms in High Risk Patients (Normalized)')

    #st.image('/Users/devgup/Documents/Documents/MBAI/KiranSimpleRAG/plots/death_count.png', caption='Death Count of High Risk Patients')
    #st.image('/Users/devgup/Documents/Documents/MBAI/KiranSimpleRAG/plots/demographics.png', caption='Demographics of High Risk Patients')
    #st.image('/Users/devgup/Documents/Documents/MBAI/KiranSimpleRAG/plots/patient_condition_trend.png', caption='Condition Trend in High Risk Patients (Normalized)')
    #st.image('/Users/devgup/Documents/Documents/MBAI/KiranSimpleRAG/plots/symptoms.png', caption='Top 10 Symptoms in High Risk Patients (Normalized)')






def Chat():
    # Initialize session state
    initialize_session_state()
    st.title("GenAI Powered Chatbot: HealthBuddy")

    # Initialize Streamlit
    #st.sidebar.title("Document Processing")

    # Hardcoded file paths
    pdf_file_paths = []

    uploaded_files = []

       # Set title of the page
    st.sidebar.title("PDF File Uploader")



    pdf_file_paths = ["/Users/devgup/Downloads/ML Output.pdf","/Users/devgup/Downloads/FAQ.pdf"]

    for file_path in pdf_file_paths:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                uploaded_files.append({"name": os.path.basename(file_path), "data": file_bytes})

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file['name'])[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file['data'])
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)


def main():


    pages = {
        "Home": home,
        "Trend Analysis": trend_analysis,
        "HealthBuddy Chat Bot": Chat
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()





if __name__ == "__main__":
    main()
''

