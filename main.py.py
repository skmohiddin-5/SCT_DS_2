import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv")


print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())


# DATA CLEANING

df["Age"] = df["Age"].fillna(df["Age"].mean())


df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df.drop("Cabin", axis=1, inplace=True)


print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# EDA VISUALIZATIONS


# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# 3. Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# 4. Age Distribution Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 5. Survival based on Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival Based on Gender")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True)

plt.title("Correlation Heatmap")
plt.show()

print("\nDATA CLEANING AND EDA COMPLETED")