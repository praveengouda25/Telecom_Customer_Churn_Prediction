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
![Churn distribution](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/Churn%20Distribution.png)
> ~26% of customers switched providers.  

### 2. Churn vs Gender  
![Churn wrt Gender](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/distributionWRTGender.PNG)
> Both genders show similar churn behavior.  

### 3. Contract Types  
![Contract distribution](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/Contract%20distribution.png)  
> Customers with **Month-to-Month contracts** are most likely to churn.  

### 4. Payment Methods  
![Payment Methods](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/payment%20methods.png)  
![Payment wrt Churn](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/payment%20ethods%20with%20respectto%20churn.PNG)  
> Customers paying via **Electronic Check** churn more frequently.  

### 5. Internet Services  
![Internet Services](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/internet%20services.PNG)  
> Customers with **Fiber Optic** services show higher churn.  

### 6. Dependents  
![Dependents](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/dependents.PNG)  
> Customers **without dependents** churn more.  

### 7. Online Security  
![Online Security](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/onlineSecurity.PNG)  
> Lack of **online security services** increases churn risk.  

### 8. Senior Citizens  
![Senior Citizen](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/seniorCitzen.PNG)  
> Senior citizens have a higher churn rate.  

### 9. Billing  
![Billing](https://github.com/praveengouda25/Telecom_Customer_Churn_Prediction/blob/main/outputs/billing.PNG)  
> Customers with **Paperless Billing** are more likely to churn.  

### 10. Tech Support  
![Tech Support](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/techSupport.PNG?raw=true)  
> Lack of **Tech Support** drives higher churn.  

### 11. Charges & Tenure  
![Monthly Charges](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/carges%20distribution.PNG?raw=true)  
![Total Charges](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/total%20charges.PNG?raw=true)  
![Tenure](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/tenure%20and%20churn.PNG?raw=true)  
> New customers and those with **high monthly charges** churn more.  

---

## 🤖 Machine Learning Models

### Models Implemented  
- Logistic Regression  
- KNN  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boost  
- Voting Classifier  

### Results after K-Fold Cross Validation  
![Model Evaluation](https://github.com/praveengouda25/Telecom-Customer-Churn-prediction/blob/main/output/Model%20evaluation.PNG?raw=true)  

> **Best Model:** Voting Classifier (~85% Accuracy).  

```python
from sklearn.ensemble import VotingClassifier
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()

eclf1 = VotingClassifier(
    estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], 
    voting='soft'
)
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
