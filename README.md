# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Shape of dataset
print("Shape of dataset:", df.shape)

# Column names
print(df.columns)

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Count plot for Survived
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Distribution of Age
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare distribution
sns.boxplot(x=df['Fare'])
plt.title("Fare Distribution")
plt.show()

# Survival based on gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival based on Gender")
plt.show()
