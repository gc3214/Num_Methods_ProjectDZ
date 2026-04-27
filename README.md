# Basket Option Pricing Project

## 1. Overview

**Goal:**  
<!-- Briefly describe what the project is doing -->
Example: Price a European basket call option on multiple equities using market-implied distributions and copula-based dependence modeling.

**Assets:**  
<!-- List assets -->
- NVDA
- MSFT
- GOOGL
- ...

**Payoff:**  
<!-- Define payoff clearly -->
\[
\max\left(\sum w_i S_i(T) - K, 0\right)
\]

---

## 2. Motivation

<!-- Why this project matters -->
- Why basket options are hard to price  
- Why Black-Scholes is insufficient  
- Why dependence modeling matters  

---

## 3. Project Structure
project_root/
│
├── basket_pricing_workbook.ipynb   # Main notebook
├── marginal_distributions.ipynb    # Part 1 exploration
├── copula.ipynb                   # Part 2 exploration
├── functions_marginal_distributions.py
├── copula.py
├── data/
│   ├── marginal_dist.csv
│   ├── simulated_prices_gaussian.csv
│   ├── simulated_prices_student.csv
│   └── copula_meta.csv
└── README.md

---

## 4. Methodology

### 4.1 Part 1 – Marginal Distributions

**What we do:**  
<!-- Describe briefly -->
- Extract implied volatility from options  
- Fit volatility smile (SABR / SVI / spline)  
- Recover risk-neutral PDF using Breeden-Litzenberger  

**Why we do it:**  
<!-- Key reasoning -->
- Market-implied distributions are more realistic than normal assumptions  

**Key Outputs:**  
- PDF, CDF for each asset  
- pricing calculator

---

### 4.2 Part 2 – Dependence Modeling (Copula)

**What we do:**  
- Estimate correlation from historical returns  
- Build inverse CDFs  
- Simulate joint distributions using:
  - Gaussian copula  
  - Student-t copula  

**Why we do it:**  
- Correlation alone is not enough  
- Need to capture tail dependence  

**Key Outputs:**  
- Simulated joint terminal prices  

---

### 4.3 Part 3 – Monte Carlo Pricing

**What we do:**  
- Simulate terminal basket values  
- Compute option payoff  
- Discount expected payoff  

**Enhancement:**  
- Apply control variate (geometric basket)  

**Why we do it:**  
- No closed-form solution for arithmetic basket  
- Reduce variance of estimates  

**Key Outputs:**  
- Option price  
- Standard error  
- Runtime comparison  

---

### 4.4 Part 4 – Pricing Calculator

**What we do:**  
- Simulate terminal basket values  
- Compute option payoff  
- Discount expected payoff  

**Enhancement:**  
- Apply control variate (geometric basket)  

**Why we do it:**  
- No closed-form solution for arithmetic basket  
- Reduce variance of estimates  

**Key Outputs:**  
- Option price  
- Standard error  
- Runtime comparison  

---


## 5. Results

### 5.1 Base Case Pricing

| Model | Price | Std Error |
|------|------|----------|
| Gaussian | <!-- fill --> | <!-- fill --> |
| Student-t | <!-- fill --> | <!-- fill --> |

---

### 5.2 Strike Sensitivity

<!-- Add table or plot -->
- Key observation:  
<!-- fill -->

---

### 5.3 Model Comparison

- Difference between models:  
<!-- fill -->

- Interpretation:  
<!-- fill -->

---

### 5.4 Correlation Sensitivity

<!-- fill -->

---

### 5.5 Monte Carlo Diagnostics

- Convergence behavior:  
<!-- fill -->

- Control variate effectiveness:  
<!-- fill -->

---

## 6. Key Insights

<!-- Bullet points -->
- Insight 1  
- Insight 2  
- Insight 3  

---

## 7. Limitations

<!-- Be honest -->
- Model assumptions  
- Data limitations  
- Simulation constraints  

---

## 8. How to Run

```bash
# install dependencies
pip install numpy pandas yfinance scipy matplotlib scikit-learn

# run notebooks
jupyter notebook