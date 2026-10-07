# PoliMi GSoM — Quantitative Finance Projects

A curated collection of group projects from my **M.Sc. in Quantitative Finance** at **GSoM – Politecnico di Milano**. Each project applies quantitative methods to a real financial problem, end to end in Python. 

## Projects

| Project | Field | What it does |
|---|---|---|
| [📈 Portfolio Optimization (Markowitz)](portfolio-optimization-markowitz/) | Asset management | Mean–variance allocation on a multi-sector S&P 500 subset: shrinkage covariance, efficient frontiers, CAPM, Black–Litterman | 
| [🤖 Statistical Arbitrage EWS (LSTM)](statistical-arbitrage-lstm-ews/) | Quant / ML | Pairs-trading Early Warning System: OPTICS + cointegration pair selection, LSTM novelty detector | 
| [📰 Financial News Sentiment (BERT + LoRA)](financial-news-sentiment-bert-lora/) | NLP | Explaining same-day stock moves from headlines via two-stage LoRA transfer learning | 
| [🌱 ESG & the Efficient Frontier](esg-markowitz-frontier/) | Sustainable finance | How an ESG-score constraint reshapes the Markowitz frontier on Russell 3000 stocks |
| [🛡️ Risk Management — VaR on a US Equity Portfolio](risk-management-var-portfolio/) | Market risk | Value-at-Risk (parametric, historical, Monte Carlo) on a seven-stock portfolio, with backtesting and diversification analysis |

Each subfolder has its own README with full methodology, results, and instructions.

## Tech across the portfolio

Python · PyTorch · scikit-learn · statsmodels · HuggingFace Transformers + PEFT (LoRA) · NumPy · pandas · SciPy · Matplotlib · seaborn

## Repository structure

```
POLIMI_GSOM/
├── portfolio-optimization-markowitz/   # Markowitz / CAPM / Black-Litterman
├── statistical-arbitrage-lstm-ews/     # Pairs trading + LSTM anomaly detection
├── financial-news-sentiment-bert-lora/ # BERT + LoRA financial NLP
├── esg-markowitz-frontier/             # ESG-constrained efficient frontier
├── risk-management-var-portfolio/      # VaR, backtesting & diversification
├── LICENSE
└── README.md
```

## Teams

These were group projects. 
- **Portfolio Optimization** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **Statistical Arbitrage EWS** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **Financial News Sentiment** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **ESG & the Efficient Frontier** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **Risk Management — VaR on a US Equity Portfolio** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
