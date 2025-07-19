import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Student_Performance_Prediction\Dataset\student_performance_dataset.csv")


# Set style
sns.set(style="whitegrid")

# Plot all numerical features against final_grade
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_features.remove('final_grade')  # exclude target

# Create individual plots
for col in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=col, y='final_grade')
    sns.regplot(data=df, x=col, y='final_grade', scatter=False, color='red')
    plt.title(f'Final Grade vs {col}')
    plt.xlabel(col)
    plt.ylabel('Final Grade')
    plt.tight_layout()
    plt.show()

