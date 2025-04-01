import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend for saving plots without displaying them
import matplotlib.pyplot as plt
from fpdf import FPDF

# Function to read data from CSV files
def read_data():
    conditions = pd.read_csv("conditions.csv")
    patients = pd.read_csv("patients.csv")
    observations = pd.read_csv("observations.csv")
    return conditions, patients, observations

# Function to get the top N most occurring condition codes
def get_top_condition_codes(conditions, top_n=3):
    top_codes = conditions['CODE'].value_counts().head(top_n).index.tolist()
    return top_codes

# Function to get patients who have the top N most occurring condition codes
def get_patients_for_top_codes(conditions, observations):
    top_codes = get_top_condition_codes(conditions)
    patients = conditions[conditions['CODE'].isin(top_codes)]['PATIENT'].unique()
    return patients

# Function to analyze symptoms of high-risk patients
def symptoms_analysis(highrisk_patient_conditions):
    symptom_table = highrisk_patient_conditions.groupby('DESCRIPTION').size().reset_index(name='COUNT').sort_values(by='COUNT', ascending=False)
    return symptom_table

# Function to analyze disease trends of high-risk patients
def disease_trend_analysis(highrisk_patient_conditions, top_n=5, min_count=100):
    condition_counts = highrisk_patient_conditions['DESCRIPTION'].value_counts()
    top_conditions = condition_counts[condition_counts >= min_count].head(top_n).index.tolist()
    filtered_trend = highrisk_patient_conditions[highrisk_patient_conditions['DESCRIPTION'].isin(top_conditions)]
    disease_trend = filtered_trend.groupby(['START', 'DESCRIPTION']).size().reset_index(name='COUNT')
    return disease_trend

# Function to analyze demographics of high-risk patients
def demographic_analysis(patients, highrisk_patient_ids):
    demographics = patients[patients.Id.isin(highrisk_patient_ids)][['RACE', 'ETHNICITY', 'GENDER']]
    race_count = demographics.groupby('RACE').size().reset_index(name='COUNT')
    ethnicity_count = demographics.groupby('ETHNICITY').size().reset_index(name='COUNT')
    gender_count = demographics.groupby('GENDER').size().reset_index(name='COUNT')
    return race_count, ethnicity_count, gender_count


# Function to analyze death statistics of high-risk patients
def death_analysis(patients, highrisk_patient_ids):
    death_data = patients[patients.Id.isin(highrisk_patient_ids)][['DEATHDATE']]
    death_data['DIED'] = death_data['DEATHDATE'].notnull()
    death_count = death_data['DIED'].value_counts().reset_index(name='COUNT')
    return death_count


# Function to visualize the top symptoms of high-risk patients
def visualize_symptoms(symptom_table, total_patients, save_path):
    symptom_table['COUNT_PERCENT'] = (symptom_table['COUNT'] / total_patients) * 100
    plt.figure(figsize=(12, 8))  # Increase the figure size
    ax = sns.barplot(x='COUNT_PERCENT', y='DESCRIPTION', data=symptom_table.head(10))
    plt.title('Top 10 Symptoms in High Risk Patients (Normalized)')
    plt.xlabel('Percentage')
    plt.ylabel('Symptom')
    plt.xticks(rotation=45)  # Rotate the y-axis labels for better readability
    # Add labels with actual count values on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.2f}%", ((p.get_width() * 1.005), p.get_y() + p.get_height() / 2), ha='left', va='center')
    plt.tight_layout()
    plt.savefig(save_path + 'symptoms.png')
    plt.close()


def visualize_condition_trend(disease_trend, save_path):
    # Calculate the total count for each date
    total_counts = disease_trend.groupby('START')['COUNT'].transform('sum')
    # Calculate the percentage of each condition count relative to the total count for each date
    disease_trend['COUNT_PERCENT'] = (disease_trend['COUNT'] / total_counts) * 100

    plt.figure(figsize=(16, 8))
    sns.lineplot(x='START', y='COUNT_PERCENT', hue='DESCRIPTION', data=disease_trend)
    plt.title('Condition Trend in High Risk Patients (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Percentage of Total Count')
    plt.xticks(rotation=45)
    plt.legend(title='Top 5 Conditions')
    plt.savefig(save_path + 'patient_condition_trend.png')
    plt.close()


# Function to visualize the demographics of high-risk patients
def visualize_demographics(race_count, ethnicity_count, gender_count, total_patients, save_path):
    race_count['COUNT_PERCENT'] = (race_count['COUNT'] / total_patients) * 100
    ethnicity_count['COUNT_PERCENT'] = (ethnicity_count['COUNT'] / total_patients) * 100
    gender_count['COUNT_PERCENT'] = (gender_count['COUNT'] / total_patients) * 100

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    ax1 = sns.barplot(x='COUNT_PERCENT', y='RACE', data=race_count)
    plt.title('Race Distribution (Normalized)')
    plt.xlabel('Percentage')
    plt.ylabel('Race')
    for p in ax1.patches:
        width = p.get_width()
        plt.text(width * 1.01, p.get_y() + p.get_height() / 2, f"{width:.2f}%", ha='left', va='center')
    ax1.set_xlim(right=100)  # Adjust x-axis limit

    plt.subplot(1, 3, 2)
    ax2 = sns.barplot(x='COUNT_PERCENT', y='ETHNICITY', data=ethnicity_count)
    plt.title('Ethnicity Distribution (Normalized)')
    plt.xlabel('Percentage')
    plt.ylabel('Ethnicity')
    for p in ax2.patches:
        width = p.get_width()
        plt.text(width * 1.01, p.get_y() + p.get_height() / 2, f"{width:.2f}%", ha='left', va='center')
    ax2.set_xlim(right=100)  # Adjust x-axis limit

    plt.subplot(1, 3, 3)
    ax3 = sns.barplot(x='COUNT_PERCENT', y='GENDER', data=gender_count)
    plt.title('Gender Distribution (Normalized)')
    plt.xlabel('Percentage')
    plt.ylabel('Gender')
    for p in ax3.patches:
        width = p.get_width()
        plt.text(width * 1.01, p.get_y() + p.get_height() / 2, f"{width:.2f}%", ha='left', va='center')
    ax3.set_xlim(right=100)  # Adjust x-axis limit

    plt.tight_layout()
    plt.savefig(save_path + 'demographics.png')
    plt.close()


# Function to visualize the death statistics of high-risk patients
def visualize_death(death_count, total_patients, save_path):
    death_count['COUNT_PERCENT'] = (death_count['COUNT'] / total_patients) * 100

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x='DIED', y='COUNT_PERCENT', data=death_count)
    plt.title('Death Count in High Risk Patients (Normalized)')
    plt.xlabel('Died')
    plt.ylabel('Percentage')
    plt.xticks([0, 1], ['No', 'Yes'])
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.savefig(save_path + 'death_count.png')
    plt.close()

def save_to_pdf(lines, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)  # Use Arial font
    for line in lines:
        # Replace bullet points with hyphens
        line = line.replace('\u2022', '-')
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

def main():
    conditions, patients, observations = read_data()

    highrisk_patient_ids = get_patients_for_top_codes(conditions, observations)

    highrisk_patient_conditions = conditions[conditions.PATIENT.isin(highrisk_patient_ids)]

    total_patients = len(highrisk_patient_ids)

    # Additional analysis: Top most occurring condition
    top_condition = highrisk_patient_conditions.groupby('DESCRIPTION').size().reset_index(name='COUNT').sort_values(by='COUNT',ascending=False).head(10)['DESCRIPTION'].tolist()

    top_condition_patients = highrisk_patient_conditions[highrisk_patient_conditions['DESCRIPTION'].isin(top_condition)]

    top_condition_trend = disease_trend_analysis(top_condition_patients)

    symptom_table = symptoms_analysis(highrisk_patient_conditions)

    race_count, ethnicity_count, gender_count = demographic_analysis(patients, highrisk_patient_ids)
    death_count = death_analysis(patients, highrisk_patient_ids)

    # Collect output into a list
    output_lines = []
    output_lines.append("\nTop Condition:")
    for condition in top_condition:
        output_lines.append(f"• {condition}")

    output_lines.append("\nTop Symptoms:")
    for index, row in symptom_table.sort_values(by='COUNT', ascending=False).head(10).iterrows():
        output_lines.append(f"{row['DESCRIPTION']:50} - {row['COUNT']}")

    output_lines.append("\nRace Distribution:")
    for index, row in race_count.sort_values(by='COUNT', ascending=False).iterrows():
        output_lines.append(f"{row['RACE']:20} - {row['COUNT']}")

    output_lines.append("\nEthnicity Distribution:")
    for index, row in ethnicity_count.sort_values(by='COUNT', ascending=False).iterrows():
        output_lines.append(f"{row['ETHNICITY']:20} - {row['COUNT']}")

    output_lines.append("\nGender Distribution:")
    for index, row in gender_count.sort_values(by='COUNT', ascending=False).iterrows():
        output_lines.append(f"{row['GENDER']:10} - {row['COUNT']}")

    output_lines.append("\nDeath Count:")
    for index, row in death_count.sort_values(by='COUNT', ascending=False).iterrows():
        output_lines.append(f"{'Died' if row['DIED'] else 'Not Died':20} - {row['COUNT']}")

    # Save output to PDF
    save_to_pdf(output_lines, "trendanalysis_output.pdf")

    save_path = './plots/'
    visualize_symptoms(symptom_table, total_patients, save_path)
    visualize_demographics(race_count, ethnicity_count, gender_count, total_patients, save_path)
    visualize_death(death_count,total_patients, save_path)

    # Visualize top condition trend
    visualize_condition_trend(top_condition_trend, save_path)

if __name__ == "__main__":
    main()
