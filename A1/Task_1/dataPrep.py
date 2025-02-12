import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# Reads the CSV file containing the customer churn data
df = pd.read_csv("customer_churn.csv")

# 2. Display descriptive statistics
# By default, df.describe() shows a statistical summary (count, mean, std, min, quartiles, max) for numeric columns.
print('\n------------------------------Displaying summary statistics------------------------------')
print(df.describe())

# 3. Identify missing values
# df.isnull().sum() returns the count of missing (NaN) values in each column.
print('\n------------------------------Identifying missing values------------------------------')
print(df.isnull().sum())

# 4. Handle missing values
#    a) Fill numeric columns with their median value
#       This helps minimize the impact of outliers compared to using the mean.
df.fillna(df.median(numeric_only=True), inplace=True)

#    b) Fill remaining missing values with the mode
#       This is common for categorical columns; the "most frequent" value replaces NaNs.
df.fillna(df.mode().iloc[0], inplace=True)

# 5. Convert categorical variables to numerical using Label Encoding
#    Most machine learning algorithms require numeric inputs rather than categorical strings.
#    LabelEncoder() maps each unique categorical value to an integer.
label_encoders = {}
categorical_cols = ["Preferred_Content_Type", "Membership_Type", "Payment_Method"]

for col in categorical_cols:
    le = LabelEncoder()
    # fit_transform learns the mapping from strings to integers and applies it to the column
    df[col] = le.fit_transform(df[col])
    # Store the encoder if you need to invert (integer -> string) later
    label_encoders[col] = le

# 6. Visualize data distributions: histograms and box plots
#    Gives a quick overview of how numeric features are distributed and if there are outliers.

print('\n------------------------------Visualizing data distributions------------------------------')

# 6a. Histograms
#    Creates a grid of subplots, each showing a histogram for one numeric column.
#    "bins=15" can be adjusted depending on how much detail you want in each histogram.
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop("CustomerID", errors="ignore")

fig, axes = plt.subplots(nrows=len(numeric_cols) // 3 + 1, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].hist(ax=ax, bins=15, edgecolor="black")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

# Hide empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Histograms of Numeric Columns", y=1.02)
plt.tight_layout()
plt.show()

# 6b. Box Plots
#    Displays box plots for each numeric column side-by-side.
#    Box plots show quartiles, median, and potential outliers.

plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.title("Box Plots of Numeric Columns", fontsize=14, pad=15)
plt.xticks(rotation=45)
plt.ylabel("Value")
plt.show()

# 7. Correlation heatmap
#    Shows pairwise correlation coefficients between numeric features.
#    "annot=True" displays the numerical correlation values in each cell.
#    "cmap='coolwarm'" sets the color palette.
print('\n------------------------------Checking for correlations between variables------------------------------')
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns=['CustomerID']).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()