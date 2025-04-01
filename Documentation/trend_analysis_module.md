# CS6440_TechnicaldoctorsProject - Trend Analysis

## Trend Analysis for FHIR Data
* Our project focuses on epidemic prediction trend analysis using synthetic data for COVID-19, generated with the Synthea utility.
* We created a sample FHIR dataset of 10,000 patients to identify prevalent diseases and patterns related to epidemics.
* Our goal is to enhance healthcare strategies and outcomes by understanding the demographic and clinical characteristics of high-risk patients.

### Data Collection:
* We collected data from CSV files using Pandas, including information on conditions, patients, and observations.

### Analysis Functions:
* Top Condition Codes: Identified the most occurring disease condition codes among high-risk patients.
* Patients for Top Codes: Found patients with the most occurring condition codes.
* Symptoms Analysis: Analyzed symptoms of high-risk patients.
* Disease Trend Analysis: Examined disease trends among high-risk patients.
* Demographic Analysis: Analyzed demographics (race, ethnicity, gender) of high-risk patients.
* Death Analysis: Studied death statistics of high-risk patients.

### Visualization Functions:
Utilized Seaborn and Matplotlib for visualization.
* Visualizing Top Symptoms: Displayed top symptoms of high-risk patients in a bar chart.
* Visualizing Condition Trend: Presented the trend of top conditions among high-risk patients over time.
* Visualizing Demographics: Used bar charts to show race, ethnicity, and gender distribution among high-risk patients.
* Visualizing Death Statistics: Illustrated death count among high-risk patients using a bar chart.

### PDF Generation:
Results were saved to a PDF file for easy sharing and reference. This PDF is also used by our Generative AI chatbot for answering user questions.

### Key Findings:
* Top Conditions: The most prevalent conditions among high-risk patients were Suspected COVID-19, COVID-19, Fever, Cough, Loss of taste, Fatigue, Obesity, Sputum finding, Prediabetes, and Anemia. These conditions were significant factors in determining the risk level of patients for epidemic prediction.
* Top Symptoms: The most common symptoms among high-risk patients were Suspected COVID-19, COVID-19, Fever, Cough, Loss of taste, Fatigue, Obesity, Sputum finding, Prediabetes, and Anemia. These symptoms played a crucial role in identifying and managing high-risk patients for epidemic prediction.
* Race Distribution: The impacted population was predominantly White (7632), followed by Black (768), Asian (649), Native (49), and Other (8). This distribution provides insights into racial disparities in health outcomes among high-risk patients for epidemic prediction.
* Ethnicity Distribution: Non-Hispanic individuals (8138) were more impacted than Hispanic individuals (968) among high-risk patients. This distribution highlights the importance of considering ethnicity in healthcare interventions and policies for epidemic prediction.
* Gender Distribution: Both females (4773) and males (4333) were significantly impacted by the top conditions. This finding underscores the importance of gender-sensitive approaches in healthcare delivery and management for epidemic prediction.
* Death Count: Out of the identified high-risk patients, 8749 did not die, while 357 died. This data emphasizes the importance of early detection, timely intervention, and effective management strategies for high-risk patients to prevent adverse outcomes for epidemic prediction.

### Conclusion:
Our analysis of trends among high-risk patients for epidemic prediction has provided valuable insights into the prevalence of Suspected COVID-19, COVID-19, and Fever, along with other key conditions and symptoms. The demographic distributions highlighted racial and ethnic disparities in healthcare outcomes, emphasizing the need for targeted interventions for epidemic prediction.

