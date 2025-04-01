# CS6440_TechnicaldoctorsProject - Covid 19 mortality risk prediction

## Problem Statement
* The goal of this is to predict covid 19 mortality risk for the patients given their profile, history of diseases, demographics, current symptoms, etc.. 
* The risk is classifed in two categories: High risk (mapped to 1) and low risk (mapped to 0).
* This will help high risk patients to take preventive measures early on. 

### Data Collection:
* We have extracted the data from FHIR and Synthea from conditions, patients, and observations table corresponding to 100 patients. These made for 12500 observation records. 

### Feature Engineering:
* We dropped irrelevant column for training like id, city, etc.
* We dropped columns with all Nan values.
* We transformed age column to a categorical column by creating 5 age buckets: ['0-30', '31-40', '41-50', '51-60', '61+']
* We transformed other categorical columns (race, ehnicity, state, etc) into numberical columns using label encoder.

### Model Training:
* We slpit the dataset into 80% training data and 20% testing data. 
* We trained random forest model with 10 trees and gini index as our split criteria.
* We training random forest model because it is an ensemble method. Each tree is trained on a subset of data and therefore it is robust to overfitting. 
* Further, we had unbalanced dataset with low risk records being the majority class, therefore random forest gives more weightage to minority class. 
* We saved our model as a joblib file under https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/tree/main/project_contents/RiskPredictionModel 


### Model Evaluation:
* We could achive 99.2% accuracy on our test datset.

### Interaction on the UI
* Users can try out our application. Follow the instructions in [user manual](https://github.gatech.edu/dgupta311/CS6440_TechnicaldoctorsProject/blob/main/Documentation/user_manual.md) for the same.

### Results:
* Results can be seen under 'ML prediction on FHIR data' page on the UI or can be queried through the Health Buddy chatbot. 
