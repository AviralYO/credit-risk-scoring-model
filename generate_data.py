import numpy as np
import pandas as pd

np.random.seed(42)
n = 30000

annual_income        = np.random.lognormal(mean=10.8, sigma=0.6, size=n).clip(20000, 500000)
loan_amount          = np.random.lognormal(mean=9.5,  sigma=0.6, size=n).clip(1000, 40000)
interest_rate        = np.random.normal(loc=13.5, scale=5.0, size=n).clip(5.0, 30.0)
employment_length    = np.random.choice(range(11), size=n, p=[0.08,0.12,0.11,0.10,0.09,0.09,0.09,0.09,0.09,0.08,0.06])
credit_history_years = np.random.normal(loc=12, scale=6, size=n).clip(1, 40)
num_open_accounts    = np.random.poisson(lam=10, size=n).clip(1, 30)
revolving_balance    = np.random.lognormal(mean=8.5, sigma=1.0, size=n).clip(0, 80000)
revolving_limit      = np.maximum(np.random.lognormal(mean=10.5, sigma=0.8, size=n).clip(500, 150000), revolving_balance * 1.1)
loan_grade_arr       = np.random.choice(['A','B','C','D','E','F','G'], size=n, p=[0.20,0.25,0.22,0.15,0.10,0.05,0.03])
delinq_2yrs          = np.random.poisson(lam=0.3, size=n).clip(0, 10)
loan_purpose         = np.random.choice(['debt_consolidation','home_improvement','medical','small_business','other'], size=n, p=[0.45,0.20,0.15,0.10,0.10])

dti                 = (loan_amount / (annual_income / 12)).clip(0, 50)
credit_utilization  = (revolving_balance / revolving_limit).clip(0, 1)
loan_to_income      = (loan_amount / annual_income).clip(0, 5)
gmap                = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
grade_num           = np.array([gmap[g] for g in loan_grade_arr])
missed_payment_flag = (delinq_2yrs > 0).astype(int)

log_odds = (
    -4.5
    + 0.15 * dti
    + 3.0  * credit_utilization
    + 0.8  * loan_to_income
    + 0.08 * interest_rate
    + 0.50 * grade_num
    + 1.5  * missed_payment_flag
    - 0.05 * employment_length
    - 0.05 * credit_history_years
    + np.random.normal(0, 0.3, n)
)
default_prob = 1 / (1 + np.exp(-log_odds))
loan_status  = (np.random.random(n) < default_prob).astype(int)
print(f"Default rate: {loan_status.mean()*100:.1f}%")

df = pd.DataFrame({
    'annual_income': annual_income, 'loan_amount': loan_amount,
    'interest_rate': interest_rate, 'employment_length': employment_length,
    'credit_history_years': credit_history_years, 'num_open_accounts': num_open_accounts,
    'revolving_balance': revolving_balance, 'revolving_limit': revolving_limit,
    'delinq_2yrs': delinq_2yrs, 'loan_grade': loan_grade_arr, 'loan_purpose': loan_purpose,
    'dti': dti, 'credit_utilization': credit_utilization, 'loan_to_income': loan_to_income,
    'missed_payment_flag': missed_payment_flag, 'loan_status': loan_status
})
df.to_csv('/home/claude/credit_risk/lending_data.csv', index=False)
print(f"Saved: {df.shape}")
