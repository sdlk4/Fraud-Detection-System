\# 🛡️ Fraud Detection System



This project is a \*\*complete fraud detection pipeline\*\* that uses machine learning to identify fraudulent transactions.  

It covers everything from \*\*exploratory data analysis (EDA)\*\* to \*\*model training, prediction, and automated alerts\*\* via \*\*email and Slack\*\*.  





\## 🚀 Features

\- \*\*Data Exploration \& Visualization\*\* → Class balance, transaction distributions.  

\- \*\*Feature Engineering\*\* → Time-based and behavioral features to improve model accuracy.  

\- \*\*Model Training\*\* → Random Forest and XGBoost classifiers.  

\- \*\*Model Saving \& Prediction\*\* → Save trained models, scalers, and metadata for reuse.  

\- \*\*Email Alert System\*\* → Get notified when fraud is detected.  

\- \*\*Slack Alert System\*\* → Team notifications through Slack bots or webhooks.  

\- \*\*End-to-End Pipeline\*\* → Automates the full fraud detection workflow.  





\## 📂 Project Structure

FRAUD/

│

├── data/ # Dataset storage

├── logs/ # Log files

├── models/ # Trained models and scalers

│

├── 1\_data\_analysis.py

├── 2\_feature\_engineering.py

├── 3\_model\_training.py

├── 4\_model\_saving\_prediction.py

├── 5\_alert\_system.py

├── 6\_slack\_alert\_system.py

├── 7\_fraud\_detection\_pipeline.py

│

├── creditcard.csv # Main dataset (download from Kaggle)

├── new\_transactions.csv # New unseen transactions for testing

├── Distributiongraphs.png # Dataset visualizations

├── Graphs.png # Model comparison and performance metrics

├── fraud\_detection.log # Sample log file

├── requirements.txt # Dependencies

└── README.md # Documentation



yaml

Copy code



---



\## 📊 Dataset

We use the \*\*Credit Card Fraud Detection Dataset\*\* from Kaggle:  

👉 \[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)



\- \*\*Total transactions\*\*: 284,807  

\- \*\*Fraudulent transactions\*\*: 492 (~0.17%)  

\- Highly \*\*imbalanced dataset\*\*  



⚠️ You must download the dataset yourself and place `creditcard.csv` into the `FRAUD/` folder.  



---



\## 📈 Exploratory Data Analysis



\### Class Distribution

!\[Class Distribution](Distributiongraphs.png)  

\- Fraud cases are extremely rare compared to normal ones.  



\### Model Performance

!\[Model Performance](Graphs.png)  

\- Both Random Forest and XGBoost achieve \*\*high ROC-AUC (~0.98)\*\*.  

\- Random Forest performs slightly better overall.  

\- Confusion matrices show very few false negatives, which is critical in fraud detection.  



---



\## 🛠️ Installation



Clone this repository:

```bash

git clone https://github.com/yourusername/FRAUD.git

cd FRAUD

Install dependencies:



bash

Copy code

pip install -r requirements.txt

▶️ Usage

Run the full pipeline

bash

Copy code

python 7\_fraud\_detection\_pipeline.py

This will:



Load and preprocess dataset



Engineer new features



Train the fraud detection model



Save model + scaler + metadata



Predict on new transactions (new\_transactions.csv)



Trigger email/Slack alerts if fraud is found



Generate visualizations and logs



Run scripts individually

Data analysis → python 1\_data\_analysis.py



Feature engineering → python 2\_feature\_engineering.py



Model training → python 3\_model\_training.py



Model saving \& prediction → python 4\_model\_saving\_prediction.py



Email alerts → python 5\_alert\_system.py



Slack alerts → python 6\_slack\_alert\_system.py



📧 Email Alerts

Requires Gmail App Password (not your real password).



Update credentials inside 5\_alert\_system.py.



Fraud alerts will be sent automatically when suspicious activity is detected.



💬 Slack Alerts

Configure with either a Bot Token or Webhook URL.



Update details inside 6\_slack\_alert\_system.py.



📊 Results

ROC-AUC: ~0.978 (Random Forest), ~0.976 (XGBoost)



Precision-Recall curves show good fraud detection performance despite imbalance.



Confusion Matrices confirm very few missed fraud cases.



Example prediction output:



Transaction ID: 105

Fraud Probability: 0.87 (87%)

Prediction: FRAUD

Email Alert: Sent

Slack Alert: Sent

