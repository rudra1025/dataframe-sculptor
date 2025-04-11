import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

class DataFrameSculptor:
    def __init__(self, df):
        """Initialize with a DataFrame"""
        self.df = df.copy()
    
    def handle_missing_values(self, strategy="mean", fill_value=None):
        """
        Handles missing values in the dataset.
        - strategy: "mean", "median", "mode", or "fill" (custom value)
        - fill_value: value to fill if strategy="fill"
        """
        if strategy == "mean":
            self.df.fillna(self.df.mean(), inplace=True)
        elif strategy == "median":
            self.df.fillna(self.df.median(), inplace=True)
        elif strategy == "mode":
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        elif strategy == "fill":
            self.df.fillna(fill_value, inplace=True)
        else:
            print("Invalid strategy! Choose from 'mean', 'median', 'mode', 'fill'.")
    
    def remove_outliers(self, method="zscore", threshold=3):
        """
        Removes outliers using Z-score or IQR method.
        - method: "zscore" or "iqr"
        - threshold: threshold for Z-score
        """
        if method == "zscore":
            z_scores = (self.df - self.df.mean()) / self.df.std()
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        
        elif method == "iqr":
            Q1 = self.df.quantile(0.25)
            Q3 = self.df.quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        else:
            print("Invalid method! Choose from 'zscore' or 'iqr'.")
    
    def encode_categorical(self, method="onehot"):
        """
        Encodes categorical features.
        - method: "onehot" (One-Hot Encoding) or "label" (Label Encoding)
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        
        if method == "onehot":
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        elif method == "label":
            label_enc = LabelEncoder()
            for col in categorical_cols:
                self.df[col] = label_enc.fit_transform(self.df[col])
        else:
            print("Invalid method! Choose from 'onehot' or 'label'.")
    
    def scale_features(self):
        """Scales numerical features using StandardScaler."""
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
    
    def remove_highly_correlated(self, threshold=0.9):
        """Removes features with high correlation to reduce dimensionality."""
        corr_matrix = self.df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
    
    def basic_statistics(self):
        """Displays basic statistical metrics."""
        return self.df.describe()
    
    def visualize_correlations(self):
        """Plots a heatmap to show feature correlations."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()
    
    def get_cleaned_data(self):
        """Returns the cleaned DataFrame."""
        return self.df

# ------------- Usage Example -----------------
# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual dataset path

# Initialize DataFrameSculptor
sculptor = DataFrameSculptor(df)

# Handle missing values
sculptor.handle_missing_values(strategy="mean")

# Remove outliers using IQR
sculptor.remove_outliers(method="iqr")

# Encode categorical features using One-Hot Encoding
sculptor.encode_categorical(method="onehot")

# Scale numerical features
sculptor.scale_features()

# Remove highly correlated features
sculptor.remove_highly_correlated(threshold=0.9)

# Show basic statistics
print(sculptor.basic_statistics())

# Visualize correlations
sculptor.visualize_correlations()

# Get the final cleaned DataFrame
cleaned_df = sculptor.get_cleaned_data()

