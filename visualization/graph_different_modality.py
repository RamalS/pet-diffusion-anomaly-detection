import pandas as pd
import matplotlib.pyplot as plt

# 1. Load CSV into a DataFrame
#    Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('../fb_metadata.csv')

# 2. Filter rows where Modality == 'PT'
pt_df = df[df['Modality'] == 'PT']

# 3. Count how many times each diagnosis appears
diagnosis_counts = pt_df['diagnosis'].value_counts()

# 4. Plot a bar chart
plt.figure(figsize=(10, 6))
diagnosis_counts.plot(kind='bar')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.title('Diagnosis Counts for Modality = PT')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
