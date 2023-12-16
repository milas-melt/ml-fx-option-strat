# Reproducing and enhancing JP Morgan FX Derivatives Research Note
## Machine Learning approach to FX option trading

"
- Artificial Intelligence (AI) has gained a lot of popularity in recent years, notably thanks to developments in hardware and computing capacity, which have allowed the treatment of larger datasets and broadened realworld applications.

- Machine Learning (ML) is the branch of AI specifically devoted to predictive analysis: given a training set of inputs Y and outputs X, find the best fitting function f such that f(Y) ≈ X. In the testing phase, apply the fit f to new inputs to make predictions.

- In this preliminary note we explore a ML – specifically Supervised Learning - approach to trade decision on 1M EUR/USD ATM options. We train standard ML models on a fairly large set of cross-asset and macro indicators to decide whether one should buy/sell/do nothing.

- Among the ML models we test, we find that k-Nearest Neighbors (kNN) and Support Vector Machines deliver the best predictive performances (>80% success rates in our implementations) when the dimensionality of the dataset is reduced through Principal Component Analysis (PCA).

- On the other hand Naïve Bayesian models and Decision Trees, which resemble the most how human experts form their decisions, fare more poorly.

- This intuitively validates the idea that dedicated ML models have the potential to outperform human discretion, especially when vast datasets are used. The framework best suited to trading decisions is to consider well defined global market states, rather than focus on a small set of determinant indicators.
"
<img width="436" alt="Screenshot 2023-12-16 at 19 26 56" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/8316b028-dfed-4ae3-858e-d9a3f10cad23">

<img width="436" alt="Screenshot 2023-12-16 at 19 27 12" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/761d3e5f-2349-425a-a6ea-f864f7166388">

<img width="436" alt="Screenshot 2023-12-16 at 19 27 29" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/46546699-5a6b-4523-aeda-3a13bb8e21ad">

<img width="436" alt="Screenshot 2023-12-16 at 19 27 44" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/9fe6d6be-ca9e-4861-a2e4-308d16209eeb">

<img width="436" alt="Screenshot 2023-12-16 at 19 27 57" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/de4d5ddb-5d93-4868-b526-2f8ebdd012d7">

<img width="391" alt="Screenshot 2023-12-16 at 19 28 19" src="https://github.com/milas-melt/ml-fx-option-strat/assets/55765976/d57cb9f5-25c1-40da-9b9b-69c71f0fb2d5">
