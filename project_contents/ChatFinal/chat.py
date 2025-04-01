from datetime import datetime

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
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
from sklearn import metrics
import time


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from reportlab.pdfgen import canvas



import chardet

output_file_name = ""
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def home():

    st.empty()
    st.title('About Technical Doctors')
    st.write('Welcome to the Website of Technical Doctors.'
             ' Our project is focused on predicting if a particular outbreak has the possibility'
             ' to turn into an epidemic or a pandemic based on existing markers from population data'
             ' from FHIR servers and other data sources. It also aims to perform analytics to find out'
             ' which segment of the population is more prone to a particular disease '
             'and will also be supported by a Gen AI-powered chatbot.')

# Add content for home page
def makeprediction():
    st.empty()
    st.sidebar.title("Upload Sample Patients Data")
    # File upload widget
    uploaded_file_1 = st.sidebar.file_uploader("Upload a Sample Patient CSV file to predict the risk of mortality due to covid", type=["csv"])

    st.sidebar.title("Upload Ground Truth CSV for Sample Patients ")
    # File upload widget
    uploaded_file_2 = st.sidebar.file_uploader("Upload a  Ground Truth CSV file to evaluate the model predictions", type=["csv"])
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        st.title("ML Prediction On FHIR DATA")
        # Load the CSV file
        X_test_5 = pd.read_csv(uploaded_file_1)
        # Load the CSV file
        #
        y_test_5 = pd.read_csv(uploaded_file_2)

        # load model
        m = os.path.join(os.path.dirname(__file__), 'RiskPredictionModel/model.joblib')
        #x_test_5 = os.path.join(os.path.dirname(__file__), 'RiskPredictionData/X_test_5patients.csv')
        #y_test_5 = os.path.join(os.path.dirname(__file__), 'RiskPredictionData/Y_test_5patients.csv')

        loaded_model = load(m)

        # load test data
        #X_test_5 = pd.read_csv(X_test_5)
        #y_test_5 = pd.read_csv(y_test_5)

        X_test_5_final = X_test_5.drop(columns=['Name'])

        # Make predictions on the test set
        y_pred = loaded_model.predict(X_test_5_final)

        #export as pdf
        y_test_df = pd.DataFrame(y_test_5)
        y_pred_df = pd.DataFrame(y_pred, columns=['pred'])

        merged_df = pd.merge(pd.merge(X_test_5, y_test_df, left_index=True, right_index=True), y_pred_df, left_index=True,
                             right_index=True)
        print(merged_df)
        # Calculate percentages
        low_risk_percentage = (merged_df['pred'] == 0).mean() * 100
        high_risk_percentage = 100 - low_risk_percentage


        output_file_name = "Upload/output1.pdf"
        c = canvas.Canvas(output_file_name)
        print(output_file_name)

        if output_file_name:

            # Create pie chart

            labels = ['Low Risk', 'High Risk']
            sizes = [low_risk_percentage, high_risk_percentage]
            explode = (0, 0.1)  # Explode the slice with high risk

            plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Save or show the plot
            plt.savefig("Plots/risk_distribution_pie_chart.png")  # Save the plot as an image

            # plt.show()  # Uncomment this line if you want to display the plot interactively
            m = os.path.join(os.path.dirname(__file__), 'Plots/risk_distribution_pie_chart.png')
            plt.clf()



             # Calculate confusion matrix
            conf_matrix = metrics.confusion_matrix(merged_df['risk'], merged_df['pred'])

            # Create a confusion matrix chart
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()

            classes = ['0', '1']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

            plt.savefig("Plots/confusion_matrix.png")  # Save the plot as an image
            # plt.show()  # Uncomment this line if you want to display the plot interactively
            n = os.path.join(os.path.dirname(__file__), 'Plots/confusion_matrix.png')
            plt.clf()

            st.image(m,'High Risk Vs Low Risk Patients Pie Chart')
            st.image(n,'High Risk Vs Low Risk Patients Confusion Matrix')
            # Create a PDF file




            # Set initial y position
            y_position = 750


            selected_columns = ['Name', 'risk', 'pred']
            df_selected = merged_df[selected_columns]
            df_renamed = df_selected.rename(columns={'Name': 'Patient Name', 'risk': 'Actual Risk', 'pred': 'Predicted Risk'})
            st.table(df_renamed)

            # Write text to the PDF
            for index, row in merged_df.iterrows():
                if (row['pred'] == 0):
                    text = "Patient " + row['Name'] + " is at low risk of mortality from Covid19"
                else:
                    text = "Patient " + row['Name'] + " is at high risk of mortality from Covid19"

                c.drawString(100, y_position, text)
                y_position -= 20

            # Save the PDF file
            c.save()
            time.sleep(5)
    else:
        st.title("Please upload test files to run the Model and display the results")

def trend_analysis():

    st.empty()
    st.title('Trend Analysis On FHIR Data')
    dc = os.path.join(os.path.dirname(__file__), 'Plots/death_count.png')
    d = os.path.join(os.path.dirname(__file__), 'Plots/demographics.png')
    pct = os.path.join(os.path.dirname(__file__), 'Plots/patient_condition_trend.png')
    s = os.path.join(os.path.dirname(__file__), 'Plots/symptoms.png')
    st.image(dc, caption='Death Count of High Risk Patients')
    st.image(d, caption='Demographics of High Risk Patients')
    st.image(pct, caption='Condition Trend in High Risk Patients (Normalized)')
    st.image(s, caption='Top 10 Symptoms in High Risk Patients (Normalized)')


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Genrative AI Chatbot using Zephyr (small LLM)"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi, I can answer all your queries regarding Covid FAQ's,Trend Analysis on FHIR Data, ML"]


def conversation_chat(query, chain, history):
    query = query + "Do not ask/answer any further questions. Do not add any extra information and give breif and concise answer"
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








def Chat():
    # Initialize session state

    st.empty()
    st.empty()

    pdf2 = os.path.join(os.path.dirname(__file__), "Upload/output1.pdf")
    print(pdf2)
    time.sleep(5)


    if os.path.exists(pdf2):
        st.title("")
        st.empty()
        st.empty()
        initialize_session_state()
        st.title("GenAI Powered Chatbot: HealthBuddy")
        st.write("Please wait HealthBuddy needs some time to be ready :)")
        st.sidebar.title("The Chat Bot: HealthBuddy")
        st.sidebar.write("The chat bot is trained on Covid FAQ's, Trend Analysis on FHIR Data Output, ML Prediction High on FHIR Data ")
        # Create a DataFrame

        st.sidebar.write("Demo Questions:")
        st.sidebar.write("  What is Covid?")
        st.sidebar.write("  What are Covid Symptoms?")
        st.sidebar.write("  How does Covid spread?")
        st.sidebar.write("  What is covid risk for Patient A? Note: so far we have Patients A to E only")
        st.sidebar.write("  Which race is most impacted by Epidemic?")

        # Hardcoded file paths
        pdf_file_paths = []

        uploaded_files = []

        pdf1 = os.path.join(os.path.dirname(__file__), 'Upload/FAQ.pdf')
        pdf3 = os.path.join(os.path.dirname(__file__), 'Upload/Trendanalysis_OutputSummary.pdf')
        pdf4 = os.path.join(os.path.dirname(__file__), 'Upload/Trendanalysis_OutputSummary_v1.pdf')
        pdf5 = os.path.join(os.path.dirname(__file__), 'Upload/Trendanalysis_OutputSummary_FAQ.pdf')



        pdf_file_paths = [pdf1,pdf4,pdf5,pdf3,pdf2]

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
            print("before embeddings")

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                               model_kwargs={'device': 'cpu'})

            print("before FAISS")

            # Create vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # Create the chain object
            print("before chain")
            chain = create_conversational_chain(vector_store)

            display_chat_history(chain)




def main():


    pages = {
        "Home": home,
        "Trend Analysis on FHIR Data": trend_analysis,
        "ML Prediction on FHIR Data": makeprediction,
        "HealthBuddy Chat Bot": Chat
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()





if __name__ == "__main__":
    main()


