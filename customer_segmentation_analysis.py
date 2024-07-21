# Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Loading the Online Retail Data
OnlineRetail = pd.read_excel('online_retail_data.xlsx', nrows = 10000)

# Dataset Information
OnlineRetail.info()

# To print Data head
OnlineRetail.head()

# To print last few data
OnlineRetail.tail()

# Checking the null vlaues
OnlineRetail.isnull().sum()

# Removing the null values and verifying again
OnlineRetail = OnlineRetail.dropna()
OnlineRetail.isnull().sum()

# Checking Duplicated Values
OnlineRetail.duplicated().sum()

# Removing Duplicated values and verifying again
OnlineRetail = OnlineRetail.drop_duplicates()
OnlineRetail.duplicated().sum()

# Checking for outliers
OnlineRetail.boxplot()

# Checking the data information after necesary preprocessing
OnlineRetail.info()

# Printing the Descriptive Statistics about the data
OnlineRetail.describe()

# Printing all columns
OnlineRetail.columns

"""Data Visualisations"""

# Data Pairplot
sns.pairplot(OnlineRetail)

# Plotting CustomerID
OnlineRetail['CustomerID'].plot(kind='hist', bins=20, title='CustomerID')
plt.gca().spines[['top', 'right',]].set_visible(False)

# Plotting UnitPrice vs CustomerID
OnlineRetail.plot(kind='scatter', x='UnitPrice', y='CustomerID', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Plotting Quantity vs UnitPrice
OnlineRetail.plot(kind='scatter', x='Quantity', y='UnitPrice', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Plotting the Correlation Matrix
CorrMatrix = OnlineRetail.select_dtypes(include=['float', 'int']).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(CorrMatrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""Data Processing for Modelling"""

# Dropping irrelevant columns for modeling
X = OnlineRetail.drop(['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'CustomerID', 'Country'], axis=1)

# Encoding the 'Country'
LabEncoder = LabelEncoder()
OnlineRetail['Country'] = LabEncoder.fit_transform(OnlineRetail['Country'])

"""# Implementation of K-means clustering"""

# Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
OnlineRetail['Cluster'] = kmeans.fit_predict(X)

# KMeans clustering analysis
KMeansClusters = OnlineRetail['Cluster'].value_counts()
AverageSilhouette = silhouette_score(X, OnlineRetail['Cluster'])
KMeansResults = (KMeansClusters, AverageSilhouette)

"""Training-Testing Split"""

# Using Quantity and UnitPrice as features, and Country as target
XClassification = X[['Quantity', 'UnitPrice']]
yClassification = OnlineRetail['Country']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(XClassification, yClassification, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
XtrainScaled = scaler.fit_transform(X_train)
XtestScaled = scaler.transform(X_test)

"""# Implementing KNN"""

# K-Nearest Neighbors Classifier
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(XtrainScaled, y_train)

# Making Predictions
yPred_KNN = KNN.predict(XtestScaled)

# Printing Accuracy
KNN_Accuracy = accuracy_score(y_test, yPred_KNN)*100
print(f"Accuracy of the KNN : {KNN_Accuracy:.2f}%")

# Printing the Confusion Matrix
KNN_Confusion = confusion_matrix(y_test, yPred_KNN)
print("Confusion Matrix:")
print(KNN_Confusion)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(KNN_Confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()

# Printing the Classification Report
KNN_Report = classification_report(y_test, yPred_KNN)
print("Classification Report: \n", KNN_Report)

"""# Implementing Random Forest"""

# Random Forest Classifier
RandForModel = RandomForestClassifier(n_estimators=100, random_state=42)
RandForModel.fit(XtrainScaled, y_train)

# Making Predictions
yPred_RandForModel = RandForModel.predict(XtestScaled)

# Priting Random Forest Accuracy
RandForModel_Accuracy = accuracy_score(y_test, yPred_RandForModel)*100
print(f"Accuracy of the Random Forest : {RandForModel_Accuracy:.2f}%")

# Printing the Confusion Matrix of the Random Forest Model
RandForModel_Confusion = confusion_matrix(y_test, yPred_RandForModel)
print("Confusion Matrix:")
print(RandForModel_Confusion)

# Printing the Classification report of the Random Forest Model
RandForModel_Report = classification_report(y_test, yPred_RandForModel)
print("Classification Report: \n", RandForModel_Report)

# plotting the models' performance
models = ['KNN', 'Random Forest']
accuracies = [KNN_Accuracy, RandForModel_Accuracy]

plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Comparison")
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
  plt.text(i, v + 2, f"{v:.2f}%", ha='center', va='bottom', color='black')
plt.show()

"""# Feature Importances"""

# Geting feature importances from the Random Forest model
Importances = RandForModel.feature_importances_

# Creating a list of feature names
Features = XClassification.columns

# Creating a DataFrame to store feature importances
FeatureImportances = pd.DataFrame({'Feature': Features, 'Importance': Importances})

# Sorting importance in descending order
FeatureImportances = FeatureImportances.sort_values(by='Importance', ascending=False)

# Printing the feature importances
print(FeatureImportances)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.barh(FeatureImportances['Feature'], FeatureImportances['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance of Random Forest")
plt.show()