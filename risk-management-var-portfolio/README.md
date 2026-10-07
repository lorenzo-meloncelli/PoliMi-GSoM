# Risk Management — Value at Risk on a US Equity Portfolio

Market-risk analysis of a seven-stock US equity portfolio: volatility estimation, Value-at-Risk under three alternative methods, out-of-sample backtesting, and a quantification of the diversification benefit.

> Academic group project — Risk Management course, GSoM (Politecnico di Milano), March 2026.

---

## Business question

Given a $100,000 portfolio built on seven diversified US equities plus a cash sleeve, how large are the potential daily losses at a 99% confidence level, and how much does sector diversification actually reduce that risk? The project applies the market-risk toolkit developed in class — volatility models, VaR methodologies and backtesting — following practices consistent with the Basel Committee's market risk framework.

## Dataset

Daily USD prices for seven US-listed stocks from 19 July 2021 to 17 November 2025 (1,131 daily observations), chosen to span four distinct sectors:

| Ticker | Company | Sector |
|--------|---------|--------|
| DHR | Danaher Corporation | Healthcare |
| VRTX | Vertex Pharmaceuticals | Healthcare |
| PANW | Palo Alto Networks | Technology |
| ZBRA | Zebra Technologies | Technology |
| HIG | Hartford Financial | Financial Services |
| NDAQ | Nasdaq Inc. | Financial Services |
| TDY | Teledyne Technologies | Industrials |

The portfolio allocates 20% each to VRTX and TDY, 10% each to the remaining five equities, and holds 10% in cash (zero return, zero covariance).

## Methodology

1. **Descriptive statistics** of daily log-returns — mean, volatility, skewness and excess kurtosis for each stock, highlighting fat tails and volatility clustering.
2. **Volatility & correlation** — simple-moving-average and EWMA (λ = 0.94) volatility estimates, plus three correlation regimes: full-sample, a 2022 stress window (Russia-Ukraine escalation and the Fed hiking cycle), and the out-of-stress remainder.
3. **Single-stock VaR** at 99%/1-day via three methods — parametric Gaussian, historical simulation, and Monte Carlo (Student-*t*, 100,000 scenarios) — calibrated on a 70% estimation window and validated out-of-sample on the remaining 30% via **Kupiec** and **Christoffersen** backtests.
4. **Portfolio construction** — three-year value path of the $100,000 investment, portfolio-level EWMA volatility, and VaR via historical simulation and the Portfolio-Normal method.
5. **Diversification benefit** — comparing the diversified variance-covariance VaR against the undiversified (perfect-correlation) VaR.
6. **Volatility dynamics** over the trailing three years.

## Selected results

- **Fat tails everywhere**: nearly every stock shows negative skew and high excess kurtosis (VRTX ≈ 30.7, PANW ≈ 27.8), so Gaussian models understate tail risk relative to historical and Student-*t* approaches.
- **Correlations rise under stress**: average pairwise correlation climbs from ≈ 0.26 (out-of-stress) to ≈ 0.48 (2022 crisis window) versus ≈ 0.33 on the full sample — diversification protection weakens exactly when it's needed most.
- **Backtesting holds up**: the Kupiec and Christoffersen tests are passed at the 5% level for six of the seven stocks; only ZBRA breaches under the Gaussian and historical methods (8 exceptions vs. ≈3.4 expected), while the Student-*t* Monte Carlo model passes cleanly.
- **Portfolio VaR (99%, 1-day, $100k)**: $2,486 (2.49%) via historical simulation vs. $2,141 (2.14%) via Portfolio-Normal — the ≈14% gap is the cost of assuming normality.
- **Diversification benefit ≈ 38.7%**: the diversified VaR ($2,198) is well below the undiversified, perfect-correlation VaR ($3,588), driven by moderate average pairwise correlation (≈0.27).
- Over the trailing three years the portfolio grew from $100,000 to ≈$148,559 (+48.6%), with clearly non-constant EWMA volatility (mean ≈14.1% annualized, with sharp spikes during market stress).

Full derivations, figures and tables are in [`report/risk_management_var_report.pdf`](report/risk_management_var_report.pdf).

## Tech stack

Python · NumPy · pandas · SciPy (`stats`, Student-*t* MLE) · statsmodels · Matplotlib · seaborn

## Repository structure

```
.
├── notebooks/
│   └── risk_management_var.ipynb   # Full analysis
├── report/
│   └── risk_management_var_report.pdf   # Written report with figures and commentary
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/risk_management_var.ipynb
```

The notebook was originally developed in Google Colab. The first cell mounts Google Drive (`from google.colab import drive`); when running locally, skip that cell and point the data-loading cells to a local path. The underlying price dataset (`Business_Case_Valore_Rischio_2026_Light.xlsx`) is course-provided and is not redistributed here.

## Team

Antonio Fontanella · Lorenzo Meloncelli · Luca Palumbo · Nicolò Ferrari

## License

Released under the MIT License — see [`LICENSE`](../LICENSE).
