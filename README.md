# CS6440_TechnicaldoctorsProject
Please commit the code for techincalDoctors project

## What do we do?
We predict if a particular outbreak has the possibility to turn into an epidemic or a pandemic based on existing markers from population data from FHIR servers and other data sources.

We also perform analytics to find out which segment of the population is more prone to a particular disease and we will provide the services of an Gen AI-powered chatbot to help users learn about these trends and the predictions better.

## Project Features :

### Trend Analysis of FHIR data :
Here we have the the trend analysis conducted on high-risk patients, focusing on the top three conditions observed in a synthetic COVID-19 dataset generated through the Synthea utility. The graphs illustrate the death counts, demographic analysis based on race, ethnicity, and gender, disease condition trends, and the top 10 symptoms observed among high-risk patients. For further insights, please refer to the accompanying video and white paper.

### ML Prediction on FHIR data :
The ML risk prediction model predicts the risk of mortality from COVID 19, given the patient's profile including history of past diseases, demographics, etc. It classifies the patient's risk in two categories: high risk and low risk. The model has been trained on data from FHIR and Synthea.

### HealthBuddy Chat Bot :
The Generative AI chat bot uses zephyr 7b model, it is trained on Covid FAQ’s, trend Analysis and ML prediction output.
The data source for all of the above was retrieved from bulk fhir data servers through a custom built pipeline where we managed to retrieve 10000 records, combine it with synthea data and clean it for the above purposes

#### Implementation Details -
Code is deployed online, link shared in our Project Submission Document.


  
