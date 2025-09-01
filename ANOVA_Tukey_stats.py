import pandas as pd
from scipy.stats import shapiro, levene
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import os

# Load the data from the Excel file
file_path = 'C:\\Users\\path_to_dir\\excel_workbook.xlsx'
sheet_name = 'Sheet1'
output_dir = 'C:\\Users\\path_to_dir\\Stats'

df = pd.read_excel(file_path, sheet_name=sheet_name)

# Drop the unnecessary column (assuming it's column C, the third column)
#df = df.drop(df.columns[2], axis=1)

# List of metrics to plot
metrics = ['Mean Distance', 'Standard Deviation']

shapiro_results = {}
anova_results = {}
tukey_results = {}
levene_results={}

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Perform one-way ANOVA for each metric
for metric in metrics:
    model = ols(f'Q("{metric}") ~ C(Q("Diagnostic category"))', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[metric] = anova_table
    
    # Check if ANOVA is significant before proceeding with Tukey's HSD
    if anova_table['PR(>F)'][0] < 0.05:
        # Perform Tukey's HSD test
        mc_results = mc.MultiComparison(df[metric], df['Diagnostic category'])
        tukey_results[metric] = mc_results.tukeyhsd()
    else:
        tukey_results[metric] = 'ANOVA not significant, no Tukey HSD test performed'

# Convert ANOVA results to DataFrame and save
anova_dfs = {metric: anova_results[metric] for metric in anova_results}
with pd.ExcelWriter(os.path.join(output_dir, 'anova_results.xlsx')) as writer:
    for metric, anova_df in anova_dfs.items():
        anova_df.to_excel(writer, sheet_name=metric)

# Save Tukey's results
tukey_dfs = {}
for metric, result in tukey_results.items():
    if isinstance(result, str):  # If ANOVA was not significant
        tukey_dfs[metric] = pd.DataFrame({'Result': [result]})
    else:
        tukey_dfs[metric] = pd.DataFrame(data=result._results_table.data[1:], 
                                         columns=result._results_table.data[0])

with pd.ExcelWriter(os.path.join(output_dir, 'tukey_results.xlsx')) as writer:
    for metric, tukey_df in tukey_dfs.items():
        tukey_df.to_excel(writer, sheet_name=metric)

# Calculate means, medians, and standard deviations for each 'Diagnostic category' (added lines)
summary_stats = df.groupby('Diagnostic category')[metrics].agg(['mean', 'median', 'std'])

# Flatten the multi-level columns
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

# Save summary statistics
summary_stats.to_excel(os.path.join(output_dir, 'summary_statistics.xlsx'))

# Function to create and save a summary plot for ANOVA results
def plot_anova_results(anova_results):
    for metric, anova_table in anova_results.items():
        print(f'ANOVA Results for {metric}')
        print(anova_table)

# Function to create and save a summary plot for Tukey's HSD results
def plot_tukey_results(tukey_results):
    for metric, result in tukey_results.items():
        if isinstance(result, str):  # If ANOVA was not significant
            print(f'Tukey HSD Results for {metric}: {result}')
        else:
            print(f'Tukey HSD Results for {metric}')
            print(result.summary())

# Create and save the summary plots
plot_anova_results(anova_results)
plot_tukey_results(tukey_results)

print("All results and plots have been saved in the specified directory.")
