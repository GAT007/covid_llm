# CS6440_TechnicalDoctors Technical Manual

* Source code : https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject

## Data source Ingestion Pipeline : 
* https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Pipelines/BulkDataExtract.ipynb

The data source ingestion pipeline is a straight forward python notebook that can be loaded onto google colab, Once there please run the following steps to get the data from the SMART BULK Data server : https://bulk-data.smarthealthit.org/?auth_type=jwks&del=NaN&dur=NaN&m=100&page=1000000&secure=0

1. Run the setup blocks on the colab file to setup the libraries. 
2. Set the access token as JWKS 
3. Generate new keys from the colab file
4. Paste the new keys into the colab file and modify the json objects as necessary
5. Send the request and that will create the csv files as necessary

### Test data sets for ML Covid 19 mortality risk prediction model: 
X_test: [https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/Pipelines/RiskPredictionData/fhir_synthea_merged_df.csv](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/X_test_5patients.csv)

y_test: https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/project_contents/RiskPredictionData/y_test_5patients.csv


### Technical Analysis on FHIR data
Technical details for this section can be found under:
https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/Documentation/trend_analysis_module.md

### ML Covid 19 mortality risk prediction model on FHIR data
Technical details for this section can be found under:
https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/Documentation/ml_risk_prediction_model.md

### Gen AI powered Chat bot
Technical details for this section can be found under:
https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/Documentation/HealthBuddy_Chatbot.md


### Deployment instructions : 
Please create an azure account and follow the instructions on https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/deployment.md

### Plots :
For the plots, please click here - https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_contents/Plots
