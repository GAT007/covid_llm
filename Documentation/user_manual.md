# CS6440_TechnicaldoctorsProject - User Manual

## Navigation Flow

Note: The flow of navigation should be Home page -> Trend Analysis -> ML Prediction -> Health Buddy Chatbot.
This is because the chatbot needs the output of trend analysis and ML prediction. Therefore, it is necessary to follow the above flow. 

1. Navigate to https://technicaldoctors.azurewebsites.net/ to access the application.
2. Home page talks about the overall project description.
3. Trend Analysis page provides the charts based on the top diseases present in the dataset and the affected patients with respect to features like demographics, death statistics and patient symptoms.
4. ML Prediction page displays the results from ML model trained to predict covid 19 mortality risk for the patients. The input to the model will be features providing information on patient's profile, history of diseases, demographics, current symptoms, etc.The risk is classifed in two categories: High risk (mapped to 1) and low risk (mapped to 0). Steps to be followed by the user on this page:
  * On the left panel, click on "Browse" button under Upload Sample Patients Data. Browse to upload [X_test_5patients.csv
](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/X_test_5patients.csv). 
  * Next, click on "Browse" button under Upload Ground Truth CSV for Sample Patients. Browse to upload [y_test_5patients.csv
](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/y_test_5patients.csv).
  * The page will display a Pie chart showing the prediction distribution, a confusion matrix showing TP, TN, FP and FN. Lastly, a table with patients ID and corresponding ground truth and model predictions for mortality risk from Covid 19.
5. HealthBuddy ChatBot should be loaded after the trend analysis and ML prediction so that the output of both are ready to be provided to the chat bot, It will take sometime (approximately 10 min) to load because of limited compute resources and cost considerations. Demo Question that can be asked to the Chatbot are provided in the left pane.


## Test data :
* Test data for ML model can be found here : [https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_c[
](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/X_test_5patients.csv)ontents/RiskPredictionData](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_contents/RiskPredictionData)

## Project Modules
1. Data ingestion : Pipeline that takes the data from the FHIR servers and processes the data in the required format.
Data pipeline -- https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Pipelines/BulkDataExtract.ipynb
2. Trend analysis -- Additional Details about the trend analysis are in https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/skalyanaraman7-Readme-Updates/Documentation/trend_analysis_module.md
Main file for executing the charts for the trend analysis -> https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/skalyanaraman7-Readme-Updates/project_contents/Pipelines/Python_TrendAnalysis/trendanalysis.py

3. ML Prediction Models

* Training the model - https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Pipelines/RiskPredictionTraining.ipynb
* Inference from the model - https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionInference.py
* Test files - https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/skalyanaraman7-Readme-Updates/project_contents/RiskPredictionData/X_test_5patients.csv, https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/y_test_5patients.csv
  
4. Gen AI Chatbot
* Model - https://drive.google.com/drive/folders/1WgbK6ed1HAh1UhA1SK23dbiK26ButlyzTest
  (This has to be placed in this location https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_contents in case one wants to deploy the model)
  * Note - The model has been stored in this location due to size constaints on GitHub and GT OneDrive. 
* The Generative AI chat bot uses zephyr 7b model, it is trained on Covid FAQ’s, trend Analysis and ML prediction output.
