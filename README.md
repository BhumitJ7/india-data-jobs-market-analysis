# Indian Data & Tech Jobs — EDA (10K Job Postings)

Exploratory data analysis of 10,000 data/tech job postings across India — covering salary, experience, seniority, location, and 27 in-demand technical skills (SQL, Python, cloud platforms, BI tools, ML frameworks, etc.)

## 📊 Dataset

`data/jobs_10k.csv` — 10,000 rows × 40 columns, including:

| Column group | Fields |
|---|---|
| Identifiers | `job_id`, `job_title`, `company` |
| Location | `city`, `state` |
| Job details | `job_type`, `date_posted`, `role_category`, `seniority` |
| Compensation | `salary_min`, `salary_max` |
| Experience | `experience_min`, `experience_max` |
| Skills (27 binary flags) | `sql`, `python`, `power_bi`, `tableau`, `excel`, `r_lang`, `dax`, `etl`, `statistics`, `git`, `looker`, `spark`, `snowflake`, `azure`, `aws`, `gcp`, `tensorflow`, `pytorch`, `kafka`, `airflow`, `dbt`, `docker`, `nosql`, `hadoop`, `sas`, `scala`, `ml` |

## 🧹 Data Cleaning

- **No duplicates or logical errors found** (`job_id` is a clean unique key; no `salary_min > salary_max` or `experience_min > experience_max` rows).
- **Salary missing in ~70% of rows** — this is structural ("not disclosed"), not an error. Rows were **kept** (not dropped) and flagged with a `salary_disclosed` boolean; salary analysis runs only on the disclosed subset.
- **Experience/seniority missing in ~8% of rows** — same treatment via `experience_disclosed` flag.
- Derived columns added: `salary_avg`, `experience_avg`, `total_skills_required`, `post_month`.
- Cleaned dataset exported to `data/jobs_10k_cleaned.csv`.

## 🔍 Key Insights

1. **Salary transparency is low** — only 29.8% of postings disclose salary, consistently across all job types.
2. **Experience drives salary far more than skill count** — correlation of 0.72 (experience) vs 0.22 (number of skills listed).
3. **Role pay tiers**: ML Engineer (₹21.99L median) > Data Engineer (₹18.97L) > Data Scientist (₹17.45L) > BI Developer (₹12.01L) ≈ Data Analyst (₹11.98L).
4. **SQL (76.9%) and Python (73.0%)** are baseline must-have skills across nearly all roles.
5. **Specialized skills carry a real premium**: TensorFlow (+₹5.2L), PyTorch (+₹3.7L), ML (+₹3.6L), Spark (+₹3.3L), Airflow (+₹3.2L) vs. postings without them.
6. **BI/reporting tools (Power BI, DAX, Excel, Tableau)** correlate with *lower* median salary — reflecting their concentration in Analyst/BI roles rather than ML/Engineering roles.
7. **Bangalore leads hiring volume** (2,209 postings) but **Gurugram pays the highest median salary** (₹16.08L), showing volume and pay aren't perfectly correlated.
8. **Skill clusters exist**: cloud/big-data tools (Spark, Kafka, Airflow, Hadoop) tend to co-occur, as do BI tools (Power BI, Tableau, DAX).

Full analysis, charts, and code in [`notebooks/EDA_Indian_Data_Jobs.ipynb`](notebooks/EDA_Indian_Data_Jobs.ipynb).

## 📁 Project Structure

```
.
├── data/
│   ├── jobs_10k.csv              # Raw dataset
│   └── jobs_10k_cleaned.csv      # Cleaned dataset with derived columns
├── notebooks/
│   └── EDA_Indian_Data_Jobs.ipynb  # Full EDA notebook
├── outputs/
│   └── *.png                     # Exported charts (12 visualizations)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Setup

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/EDA_Indian_Data_Jobs.ipynb
```

## 🛠️ Tools Used

Python · pandas · numpy · matplotlib · seaborn · Jupyter

## 📈 Possible Extensions

- Salary prediction model (regression) using experience, role, city, and skills as features
- Cluster job postings into "skill archetypes" (k-modes / PCA on skill flags)
- Interactive dashboard (Streamlit / Power BI) on top of the cleaned dataset
