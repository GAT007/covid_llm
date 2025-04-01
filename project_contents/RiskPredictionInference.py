import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from reportlab.pdfgen import canvas

def makeprediction():
    # load model
    loaded_model = load('RiskPrediction/model.joblib')

    # load test data
    X_test_5 = pd.read_csv("RiskPrediction/X_test_5patients.csv")
    y_test_5 = pd.read_csv("RiskPrediction/y_test_5patients.csv")

    X_test_5_final = X_test_5.drop(columns=['Name'])

    # Make predictions on the test set
    y_pred = loaded_model.predict(X_test_5_final)

    #export as pdf
    y_test_df = pd.DataFrame(y_test_5)
    y_pred_df = pd.DataFrame(y_pred, columns=['pred'])

    merged_df = pd.merge(pd.merge(X_test_5, y_test_df, left_index=True, right_index=True), y_pred_df, left_index=True,
                         right_index=True)

    # Create a PDF file
    pdf_filename = "output.pdf"
    c = canvas.Canvas(pdf_filename)

    # Set initial y position
    y_position = 750

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

