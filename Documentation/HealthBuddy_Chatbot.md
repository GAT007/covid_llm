**CS6440_TechnicaldoctorsProject - Health Buddy ChatBot**

**Gen AI Powered Chatbot: HealthBuddy**

Our project focuses on epidemic prediction trend analysis using synthetic data for COVID-19 and prediction of mortality risk on FHIR data and data generated with the Synthea utility and FHIR Data. The Generative AI chat bot uses zephyr 7b model, it is trained on Covid FAQ’s, trend Analysis and ML prediction output.
**NOTE:** Due to limitations of compute resources being expensive on Azure deployment, the chat bot will run slow and can take approximately 10 minutes to load, please be patient and if you want to see its working faster please refer to the project video.

**PDF Data for the ChatBot:**
The Generative AI ChatBot is trained on:

1. **Covid FAQ:** https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Upload/FAQ.pdf

2. **Trend Analysis Output:** https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Upload/Trendanalysis_OutputSummary.pdf
https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Upload/Trendanalysis_OutputSummary_FAQ.pdf
https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Upload/Trendanalysis_OutputSummary_v1.pdf

3. **ML Prediction Output:** The ML prediction output is provided realtime but we have provided a link of one of the output files generated during our implementation: https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Upload/output_2024-04-04%2008_49_01.pdf

**Model Used:**


Model - https://drive.google.com/drive/folders/1WgbK6ed1HAh1UhA1SK23dbiK26ButlyzTest (This has to be placed in this location https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_contents in case one wants to deploy the model)
Note - The model has been stored in this location due to size constaints on GitHub and GT OneDrive.
The Generative AI chat bot uses zephyr 7b model, it is trained on Covid FAQ’s, trend Analysis and ML prediction output.
The model is tuned by performing hyperparameter tuning, which involved changing the temprature value and the batch value and chunk value to receive the right results.

**UI:**

The ChatBot UI is using streamlit and the streamlit_chat from message to provide a chat flow.


**Conclusion:**


The Generative AI chatbot provides a collaboration for all the modules in our project and gives a good QnA endpoint for the the end users to ask questions and get answers in a human way. 
This chatbot in realtime scenarios can reduce the pressure on public helplines and provide users a feel of interacting with a realtime chatbot with all latest information on the pandemic/epidemic.
