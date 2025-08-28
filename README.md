<div align="right">
  
[1]: https://github.com/praveengouda25
[2]: https://www.linkedin.com/in/praveen-kumar-bcc2525/

[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]

</div>


# <div align="center">Telecom Customer Churn Prediction</div>

![Customer Churn](https://img.freepik.com/free-vector/customer-churn-rate-concept-illustration_114360-7967.jpg)  


## 📌 Introduction
Customer churn refers to when subscribers stop using a company’s services and switch to competitors.  
In the telecom industry, churn rates can be as high as **15–25% annually** due to intense competition.  

- Retaining existing customers is **more cost-effective** than acquiring new ones.  
- Machine Learning helps by **identifying at-risk customers early**, allowing targeted retention strategies.  
- This project uses ML models to predict churn and highlight factors influencing customer decisions.  

---

## 🎯 Objectives
- Calculate the **% of churn vs retained customers**.  
- Analyze **key drivers** responsible for churn.  
- Train ML models to classify **churn vs non-churn customers**.  
- Recommend the **best-performing model** for real-world deployment.  

---

## 📂 Dataset
We used the [Telco Customer Churn dataset](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data).  

**Features include:**  
- **Churn status** – Whether the customer left within the last month.  
- **Services** – Phone, internet, streaming, online security, tech support, etc.  
- **Account info** – Contract type, billing, monthly & total charges, tenure.  
- **Demographics** – Gender, senior citizen, dependents, partners.  

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Deployment:** Flask  
- **Visualization:** Matplotlib / Seaborn  

---

## 🔎 Workflow
1. **Data Preprocessing** – Handle missing values, encode categorical data, scale features.  
2. **EDA** – Analyze churn distribution & customer behavior.  
3. **Feature Engineering** – Extract meaningful variables.  
4. **Model Training** – Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, Ensemble methods.  
5. **Evaluation** – Accuracy, Precision, Recall, F1-score, ROC-AUC.  
6. **Deployment** – Flask app for churn prediction.  

---

## 📊 Exploratory Data Analysis (EDA)

### 1. Churn Distribution  
![Churn Distribution](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/Churn%20Distribution.png?raw=true)  

### 2. Contract Types  
![Contract Distribution](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/Contract%20distribution.png?raw=true)  

### 3. Paperless Billing  
![Paperless Billing](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/billing.PNG?raw=true)  

### 4. Monthly Charges Distribution  
![Monthly Charges](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/carges%20distribution.PNG?raw=true)  

---

## 🤖 Model Performance & Evaluation

### Accuracy Score Comparison  
![Accuracy Score Comparison](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/Accuracy%20score%20comparison.PNG?raw=true)  

### ROC Graph (Adaboost Example)  
![ROC Graph](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/Adaboost.PNG?raw=true)  

### Confusion Matrices  
![Confusion Matrices](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/confusion_matrix_models.PNG?raw=true)  

---

## ✅ Results
- **Best Model:** Voting Classifier (~85% Accuracy).  
- **Key Insights:**  
  - Customers with **Month-to-Month contracts** churn the most.  
  - **Electronic Check payments** are strongly linked to churn.  
  - Customers with **high monthly charges** and **short tenure** churn more.  
  - Lack of **online security, tech support, or dependents** increases churn probability.  

---

## 📌 Conclusion
This project demonstrates how **Machine Learning can effectively predict telecom customer churn**.  

- Using multiple ML models, the **Voting Classifier** performed best, achieving ~85% accuracy.  
- The analysis shows that **contract type, billing method, charges, and service features** play the biggest role in customer retention.  
- These insights can help telecom companies design **targeted retention strategies** such as:  
  - Incentives for customers on month-to-month contracts.  
  - Better support for customers using electronic checks.  
  - Discounts or loyalty benefits for high-charge and short-tenure customers.  

📈 In summary: **Predicting churn early saves costs, boosts retention, and ensures long-term customer loyalty.**  

---

## ⚙️ Usage
```bash
# Clone the repository
git clone https://github.com/praveengouda25/Telecom-Customer-Churn-prediction.git

# Navigate to project
cd Telecom-Customer-Churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
