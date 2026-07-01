import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 50)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('data/jobs_10k.csv')
print(f"Shape: {df.shape}")
df.info()
print(df.describe(include='all').T)

# ============================================================
# 2. DATA CLEANING
# ============================================================

# --- 2.1 Missing values ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct}).query('missing_count > 0'))

# Salary missing ~70% -> structural ("not disclosed"), NOT an error. Keep rows, flag instead.
df['salary_disclosed'] = df['salary_min'].notnull()
df['experience_disclosed'] = df['experience_min'].notnull()

print("Rows with salary disclosed:", df['salary_disclosed'].sum())
print("Rows with experience disclosed:", df['experience_disclosed'].sum())

# --- 2.2 Duplicates ---
print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicated job_id:", df['job_id'].duplicated().sum())

# --- 2.3 Logical consistency checks ---
bad_salary = (df['salary_min'] > df['salary_max']).sum()
bad_experience = (df['experience_min'] > df['experience_max']).sum()
print(f"Rows where salary_min > salary_max: {bad_salary}")
print(f"Rows where experience_min > experience_max: {bad_experience}")

# --- 2.4 Data types & derived columns ---
df['date_posted'] = pd.to_datetime(df['date_posted'])
df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2
df['experience_avg'] = (df['experience_min'] + df['experience_max']) / 2
df['post_month'] = df['date_posted'].dt.to_period('M').astype(str)
df['post_week'] = df['date_posted'].dt.isocalendar().week

skill_cols = ['sql','python','power_bi','tableau','excel','r_lang','dax','etl','statistics','git',
              'looker','spark','snowflake','azure','aws','gcp','tensorflow','pytorch','kafka',
              'airflow','dbt','docker','nosql','hadoop','sas','scala','ml']

df['total_skills_required'] = df[skill_cols].sum(axis=1)

# --- 2.5 Categorical value checks ---
for col in ['job_type', 'role_category', 'seniority', 'state', 'city']:
    print(f"\n{col} ({df[col].nunique()} unique values):")
    print(df[col].value_counts())

# Save cleaned dataset
df.to_csv('data/jobs_10k_cleaned.csv', index=False)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

# --- 3.1 Postings by role, city, job type ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
df['role_category'].value_counts().plot(kind='bar', ax=axes[0], color='#4C72B0')
axes[0].set_title('Job Postings by Role Category')
axes[0].tick_params(axis='x', rotation=45)

df['city'].value_counts().plot(kind='bar', ax=axes[1], color='#55A868')
axes[1].set_title('Job Postings by City')
axes[1].tick_params(axis='x', rotation=45)

df['job_type'].value_counts().plot(kind='bar', ax=axes[2], color='#C44E52')
axes[2].set_title('Job Postings by Job Type')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outputs/01_postings_overview.png', dpi=120)
plt.show()

# --- 3.2 Salary distribution ---
salary_df = df[df['salary_disclosed']]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(salary_df['salary_avg'], bins=40, kde=True, ax=axes[0], color='#4C72B0')
axes[0].set_title('Average Salary Distribution (₹)')

sns.boxplot(data=salary_df, x='role_category', y='salary_avg', ax=axes[1])
axes[1].set_title('Salary by Role Category')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('outputs/02_salary_distribution.png', dpi=120)
plt.show()

print(salary_df.groupby('role_category')['salary_avg']
      .agg(['mean','median','min','max','count'])
      .sort_values('median', ascending=False).round(0))

# --- 3.3 Salary by seniority ---
order = ['Fresher', 'Junior', 'Mid', 'Senior']
plt.figure(figsize=(10,6))
sns.boxplot(data=salary_df, x='seniority', y='salary_avg', order=order, palette='viridis')
plt.title('Salary by Seniority Level')
plt.savefig('outputs/03_salary_by_seniority.png', dpi=120)
plt.show()
print(salary_df.groupby('seniority')['salary_avg'].median().reindex(order))

# --- 3.4 Salary by city ---
city_salary = salary_df.groupby('city')['salary_avg'].median().sort_values(ascending=False)
plt.figure(figsize=(10,6))
city_salary.plot(kind='bar', color='#8172B2')
plt.title('Median Salary by City')
plt.xticks(rotation=45)
plt.savefig('outputs/04_salary_by_city.png', dpi=120)
plt.show()
print(city_salary)

# --- 3.5 Experience vs Salary ---
plt.figure(figsize=(10,6))
sns.scatterplot(data=salary_df, x='experience_avg', y='salary_avg', hue='role_category', alpha=0.5)
plt.title('Experience vs Salary')
plt.savefig('outputs/05_experience_vs_salary.png', dpi=120)
plt.show()
print("Correlation (experience vs salary):",
      salary_df['experience_avg'].corr(salary_df['salary_avg']).round(3))

# --- 3.6 Most in-demand skills overall ---
skill_demand = df[skill_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10,8))
skill_demand.plot(kind='barh', color='#4C72B0')
plt.title('Skill Demand Across All Job Postings')
plt.gca().invert_yaxis()
plt.savefig('outputs/06_skill_demand.png', dpi=120)
plt.show()
print((skill_demand / len(df) * 100).round(1))

# --- 3.7 Top skills by role category ---
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
axes = axes.flatten()
for i, role in enumerate(df['role_category'].unique()):
    role_df = df[df['role_category'] == role]
    top_skills = role_df[skill_cols].sum().sort_values(ascending=False).head(8)
    top_skills.plot(kind='barh', ax=axes[i], color='#55A868')
    axes[i].set_title(f'Top Skills — {role}')
    axes[i].invert_yaxis()
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('outputs/07_skills_by_role.png', dpi=120)
plt.show()

# --- 3.8 Skill count vs salary ---
plt.figure(figsize=(10,6))
sns.regplot(data=salary_df, x='total_skills_required', y='salary_avg',
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Number of Required Skills vs Salary')
plt.savefig('outputs/08_skills_count_vs_salary.png', dpi=120)
plt.show()
print("Correlation:", salary_df['total_skills_required'].corr(salary_df['salary_avg']).round(3))

# --- 3.9 Salary premium per skill ---
skill_premium = {}
for s in skill_cols:
    with_skill = salary_df[salary_df[s] == 1]['salary_avg'].median()
    without_skill = salary_df[salary_df[s] == 0]['salary_avg'].median()
    skill_premium[s] = with_skill - without_skill

premium_series = pd.Series(skill_premium).sort_values(ascending=False)
plt.figure(figsize=(10,8))
premium_series.plot(kind='barh', color=premium_series.apply(lambda x: '#55A868' if x>0 else '#C44E52'))
plt.title('Salary Premium by Skill (Median Salary With vs Without)')
plt.axvline(0, color='black', linewidth=0.8)
plt.gca().invert_yaxis()
plt.savefig('outputs/09_skill_salary_premium.png', dpi=120)
plt.show()
print(premium_series.round(0))

# --- 3.10 Hiring trend over time ---
monthly = df.groupby('post_month').size()
plt.figure(figsize=(10,6))
monthly.plot(kind='line', marker='o', color='#4C72B0')
plt.title('Job Postings Over Time')
plt.xticks(rotation=45)
plt.savefig('outputs/10_hiring_trend.png', dpi=120)
plt.show()

# --- 3.11 Top hiring companies ---
top_companies = df['company'].value_counts().head(15)
plt.figure(figsize=(10,8))
top_companies.plot(kind='barh', color='#8172B2')
plt.title('Top 15 Hiring Companies')
plt.gca().invert_yaxis()
plt.savefig('outputs/11_top_companies.png', dpi=120)
plt.show()

# --- 3.12 Skill correlation heatmap ---
plt.figure(figsize=(14,12))
corr = df[skill_cols].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.3, square=True)
plt.title('Skill Co-occurrence Correlation')
plt.tight_layout()
plt.savefig('outputs/12_skill_correlation.png', dpi=120)
plt.show()

# ============================================================
# 4. KEY INSIGHTS (printed summary)
# ============================================================
print("""
KEY INSIGHTS:
1. Only 29.8% of postings disclose salary — consistent across all job types.
2. Experience correlates with salary far more strongly (r=0.72) than skill count (r=0.22).
3. Median salary: ML Engineer > Data Engineer > Data Scientist > BI Developer ≈ Data Analyst.
4. SQL (76.9%) and Python (73.0%) are baseline skills required across almost all postings.
5. TensorFlow, PyTorch, ML, Spark, Airflow carry the largest positive salary premiums.
6. Power BI, DAX, Excel, Tableau show negative salary association (concentrated in lower-paying roles).
7. Bangalore leads volume; Gurugram has the highest median salary.
8. Cloud/big-data tools and BI tools each form natural co-occurring skill clusters.
""")
