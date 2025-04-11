import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandasql as psql

class DataFrameSculptor:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self, strategy="mean", fill_value=None):
        if strategy == "mean":
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        elif strategy == "median":
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        elif strategy == "mode":
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        elif strategy == "fill":
            self.df.fillna(fill_value, inplace=True)
        else:
            print("Invalid strategy! Choose from 'mean', 'median', 'mode', 'fill'.")
    
    def remove_outliers(self, method="zscore", threshold=3):
        if method == "zscore":
            z_scores = (self.df - self.df.mean(numeric_only=True)) / self.df.std(numeric_only=True)
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        elif method == "iqr":
            Q1 = self.df.quantile(0.25, numeric_only=True)
            Q3 = self.df.quantile(0.75, numeric_only=True)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            print("Invalid method! Choose from 'zscore' or 'iqr'.")
    
    def encode_categorical(self, method="onehot"):
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        if method == "onehot":
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        elif method == "label":
            label_enc = LabelEncoder()
            for col in categorical_cols:
                self.df[col] = label_enc.fit_transform(self.df[col])
        else:
            print("Invalid method! Choose from 'onehot' or 'label'.")
    
    def scale_features(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
    
    def remove_highly_correlated(self, threshold=0.9):
        corr_matrix = self.df.corr(numeric_only=True).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
    
    def basic_statistics(self):
        return self.df.describe()
    
    def visualize_correlations(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()
    
    def get_cleaned_data(self):
        return self.df

    def sql_query(self, query):
        try:
            return psql.sqldf(query, {"df": self.df})
        except Exception as e:
            print("SQL Error:", e)
            return None


# ------------------ USAGE IN COLAB ------------------

# Load dataset
df = pd.read_csv("your_dataset.csv")  # replace with your actual file path

sculptor = DataFrameSculptor(df)

# Basic cleaning steps
sculptor.handle_missing_values()
sculptor.remove_outliers()
sculptor.encode_categorical()
sculptor.scale_features()
sculptor.remove_highly_correlated()

# View basic stats
print(sculptor.basic_statistics())

# Optional: Correlation Heatmap
sculptor.visualize_correlations()

# OPTIONAL SQL QUERY FROM USER
user_query = input("Enter an SQL query to run on the cleaned DataFrame (or press Enter to skip):\n")
if user_query.strip():
    sql_result = sculptor.sql_query(user_query)
    print("SQL Query Result:")
    print(sql_result)

# Get the final cleaned data
cleaned_df = sculptor.get_cleaned_data()
