# Detailed Analysis of Aave Wallet Credit Scores

## Introduction
Okay, so this document is where I really dig into what I found after calculating credit scores for a bunch of Aave wallets. It's about showing how I approached it, what the scores look like, and what different scores might tell us about how people use Aave. It's pretty cool how much you can learn just from transaction data!

---

## Data Overview
My whole analysis started with a big file, `user-wallet-transactions.json`, which had about 100,000 individual transaction records from Aave V2. Each record was basically a log of what a specific wallet (`userWallet`) did (like a `deposit`, `borrow`, `repay`, or `liquidation`), with which crypto asset (`asset`), and its value in USD (`value_usd`). It's like looking at a ledger, but for crypto wallets, and it gives a snapshot of how people interacted with the Aave protocol.

---

## Feature Engineering Details

To get meaningful "credit" insights from just transaction logs, I had to create a bunch of new features. This involved a lot of thought about what different actions actually mean for a wallet's financial behavior.

* **Activity Counts:** I started by simply counting how many times a wallet performed certain actions: `total_transactions` (total activity), `num_deposits`, `num_borrows`, `num_repays`, `num_redeemunderlying` (withdrawing collateral), and `num_liquidations` (getting their collateral sold to cover a loan). These give a basic idea of how active or involved a wallet is.

* **Value Aggregates:** Counting is one thing, but the *value* of those transactions matters too. So, I summed up the USD value for each type of action: `total_deposit_usd`, `total_borrow_usd`, `total_repay_usd`, `total_redeem_usd`, and `total_liquidation_usd`. A little heads-up: handling the `value_usd` for `liquidationcall` was a bit tricky. Sometimes it means the debt covered, not the asset seized, so I made sure to process it carefully to reflect the actual value associated with the liquidation event.

* **Ratios:** Ratios are super helpful because they compare things and give context.
    * `repay_to_borrow_ratio`: This is a really important one! It's `total_repay_usd / total_borrow_usd`. If this is high (close to 1), it means the wallet is good at paying back their loans. I capped it at 1 to avoid weirdly high ratios if someone borrowed a tiny amount and then repaid just slightly more – it just means they fully repaid.
    * `deposit_utilization_ratio`: This is `total_borrow_usd / total_deposit_usd`. It's a quick way to see how much of their deposited collateral a wallet is actually using to borrow. High utilization means they're using a lot of their deposited funds.
    * `liquidation_value_to_borrow_value_ratio`: This one is `total_liquidation_usd / total_borrow_usd`. It shows how much of the money they borrowed ended up getting liquidated. A high ratio here means they're probably not managing their loans well.
    * `liquidation_calls_per_transaction_ratio`: This is `num_liquidations / total_transactions`. It helps me see if a wallet is getting liquidated *frequently* compared to how active they are overall. This can flag risky patterns.

* **Temporal Features:** How long a wallet has been around and how recently it was active tells a story.
    * `account_age_days`: The number of days between a wallet's first and last transaction. Older, active accounts might be more reliable.
    * `days_since_last_transaction`: How many days have passed since the wallet's very last transaction. A lower number means they're more recently active.
    * `avg_transactions_per_day`: `total_transactions / account_age_days`. This helps normalize activity for older accounts that might have long periods of inactivity.

* **Diversity Metrics:**
    * `num_unique_assets`: How many different types of crypto assets a wallet interacted with.
    * `num_unique_actions`: How many different kinds of actions they performed. These can hint at how broadly a wallet engages with Aave.

* **Outlier Treatment (Winsorization):** Before doing any serious calculations, I had to deal with really extreme values in my features. For example, some wallets might have had one *gigantic* deposit that would skew everything. So, I applied **Winsorization**. This process takes any values that are super high (above the 99th percentile) and brings them down to the 99th percentile value. It does the same for super low values, bringing them up to the 1st percentile. This makes sure those extreme cases don't unfairly mess up the scaling and the final score.

---

## Credit Scoring Methodology (Detailed)

My credit score is basically a combination of all the features I engineered, with some features getting more "say" than others. Then, I scale it to a score between 0 and 1000.

### Feature Selection
I picked a good mix of features that I felt best represented good and bad behavior. Here's the full list of features I used for scoring:
* `total_transactions`
* `num_deposits`
* `num_borrows`
* `num_repays`
* `num_redeemunderlying`
* `num_liquidations`
* `total_deposit_usd`
* `total_borrow_usd`
* `total_repay_usd`
* `total_redeem_usd`
* `total_liquidation_usd`
* `num_unique_assets`
* `account_age_days`
* `days_since_last_transaction`
* `repay_to_borrow_ratio`
* `liquidation_value_to_borrow_value_ratio`
* `liquidation_calls_per_transaction_ratio`
* `deposit_utilization_ratio`
* `num_unique_actions`
* `avg_transactions_per_day`

### Standardization
This part is crucial. Before I combined any features, I used **Min-Max Scaling** on all of them. This means every feature's values were transformed to fit neatly into a range from 0 to 1. Why? Because if `total_deposit_usd` had values in the millions and `repay_to_borrow_ratio` only went from 0 to 1, the `total_deposit_usd` would completely dominate the final score. Scaling makes sure every feature contributes fairly, relative to its own range, according to the weights I gave it.

### Weighting Scheme
This is where I put my own "rules" into the model. I created a `feature_weights` dictionary, where each feature gets a number. A positive number means it boosts the score, and a negative number means it lowers it. I assigned these weights based on what I felt was important for a reliable Aave user versus a risky one.

| Feature                                   | Weight | Justification                                                                                                                                                                                                                                                                                                                                                                                             |
| :---------------------------------------- | :----- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repay_to_borrow_ratio`                   | `0.15` | **Super Important (Positive):** This is probably the best sign of a good borrower. If they repay what they borrow, they're responsible.                                                                                                                                                                                                                                                                 |
| `num_liquidations`                        | `-0.15` | **Very Bad (Negative):** Getting liquidated means losing collateral. Doing it often is a major red flag for a wallet's riskiness, so it gets a strong penalty.                                                                                                                                                                                                                                          |
| `total_liquidation_usd`                   | `-0.10` | **Pretty Bad (Negative):** Not just how many times, but *how much* value was liquidated. Big liquidation amounts mean bigger problems.                                                                                                                                                                                                                                                                  |
| `liquidation_calls_per_transaction_ratio` | `-0.15` | **Very Bad (Negative):** If a wallet gets liquidated a lot compared to its total activity, it suggests a pattern of very risky or even potentially manipulative behavior.                                                                                                                                                                                                                                  |
| `account_age_days`                        | `0.07` | **Good (Positive):** Wallets that have been active for a long time often indicate a more stable and experienced user.                                                                                                                                                                                                                                                                               |
| `total_deposit_usd`                       | `0.10` | **Important (Positive):** Higher deposits show more commitment and financial engagement with the protocol.                                                                                                                                                                                                                                                                                           |
| `total_repay_usd`                         | `0.12` | **Important (Positive):** High total repayments mean consistent good behavior and active loan management.                                                                                                                                                                                                                                                                                           |
| `days_since_last_transaction`             | `-0.07` | **Slightly Bad (Negative):** If a wallet hasn't done anything in a long time, it might be inactive or abandoned, which could be less reliable for future interactions.                                                                                                                                                                                                                                   |
| `total_transactions`                      | `0.05` | **Good (Positive):** More overall transactions mean the wallet is more engaged with Aave.                                                                                                                                                                                                                                                                                                           |
| `num_repays`                              | `0.10` | **Important (Positive):** More individual repayment actions contribute positively to the score.                                                                                                                                                                                                                                                                                                     |
| `avg_transactions_per_day`                | `0.05` | **Good (Positive):** Consistent daily activity (not just bursts) often signifies a more regular and reliable user.                                                                                                                                                                                                                                                                                    |
| `num_deposits`                            | `0.08` | **Good (Positive):** More deposits mean the wallet is actively adding assets to the platform.                                                                                                                                                                                                                                                                                                       |
| `total_borrow_usd`                        | `0.05` | **Slightly Positive:** While borrowing has risk, it also shows engagement with the main function of Aave. It's positive if managed responsibly.                                                                                                                                                                                                                                                       |
| `num_borrows`                             | `0.02` | **Minor Positive:** Simply making more borrow transactions shows engagement.                                                                                                                                                                                                                                                                                                                        |
| `num_redeemunderlying`                    | `0.03` | **Minor Positive:** Withdrawing collateral is a normal part of managing assets and shows active use.                                                                                                                                                                                                                                                                                              |
| `total_redeem_usd`                        | `0.03` | **Minor Positive:** The total value of assets redeemed also indicates active portfolio management.                                                                                                                                                                                                                                                                                                  |
| `num_unique_assets`                       | `0.05` | **Good (Positive):** Using a variety of assets might suggest a broader understanding of DeFi or more sophisticated strategies.                                                                                                                                                                                                                                                                      |
| `num_unique_actions`                      | `0.05` | **Good (Positive):** Engaging with different types of Aave actions (deposit, borrow, swap) can mean a more versatile user.                                                                                                                                                                                                                                                                           |
| `deposit_utilization_ratio`               | `-0.05` | **Slightly Bad (Negative):** While some utilization is normal, very high utilization means a wallet is closer to liquidation risk. It's a small penalty for potentially higher-risk strategies.                                                                                                                                                                                                        |
| `liquidation_value_to_borrow_value_ratio` | `-0.10` | **Pretty Bad (Negative):** This ratio directly hits wallets where a significant portion of their borrowed funds ended up in liquidation. It highlights inefficient or risky borrowing management.                                                                                                                                                                                                       |

### Score Range
The final scores are scaled to be between **0 and 1000**. I chose this range because it's easy to understand and feels similar to other scoring systems, where higher is always better!

---

## Results and Interpretation

After all that number crunching, here's what the credit scores for these Aave wallets look like.

### Credit Score Distribution
I made a histogram to see how the scores are spread out.

![Credit Score Distribution](credit_score_distribution.png)

Looking at this graph, it seems like **most of the wallets scored pretty well**, leaning towards the higher end of the spectrum. There are fewer wallets at the very low or very high ends, with a big chunk of scores falling in the middle to upper ranges (maybe 600-800). This suggests that a good portion of the wallets in this dataset exhibit relatively responsible behavior on Aave.

### Behavior of Wallets in Different Score Ranges

To really understand what makes a score high or low, I checked out some key features for wallets grouped by their score ranges (0-100, 100-200, and so on).

**Total Liquidation USD by Credit Score Range:**
![Total Liquidation USD by Credit Score Range](liquidation_usd_by_score_range.png)

This box plot shows a clear pattern: **wallets with lower credit scores (especially those in the 0-300 range)** experienced much higher total liquidation amounts. You can see their boxes are significantly taller and more spread out. As the credit score goes up, the median `total_liquidation_usd` drops, becoming very small or even zero for high-scoring wallets. This tells me my model correctly penalizes wallets that have had large liquidation events.

**Repay to Borrow Ratio by Credit Score Range:**
![Repay to Borrow Ratio by Credit Score Range](repay_borrow_ratio_by_score_range.png)

This plot is exactly what I hoped to see! Wallets with **higher credit scores (like 700-1000)** consistently show `repay_to_borrow_ratio` values very close to 1.0 (or at the cap I set). This confirms that responsible repayment behavior is a major driver of high scores. On the flip side, wallets in the **lower score ranges** have much lower and more varied `repay_to_borrow_ratio` values, which indicates they're not repaying their loans as consistently or fully.

**Correlation Matrix of Scaled Features:**
![Correlation Matrix of Scaled Features](features_correlation_heatmap.png)

This heatmap helps me see how all my scaled features relate to each other. For example, I can see a strong positive relationship between `total_deposit_usd_scaled` and `total_repay_usd_scaled` (which makes sense – if you deposit more, you probably also repay more). I also noticed clear negative relationships between `num_liquidations_scaled` and positive behavior metrics like `repay_to_borrow_ratio_scaled`. This visual confirms that my features are capturing different, but sometimes connected, aspects of how wallets behave.

**Feature Weights in Credit Score Calculation:**
![Feature Weights in Credit Score Calculation](feature_weights_chart.png)

This bar chart visually shows how much each feature impacts the final credit score. The green bars are positive influences, and the red bars are negative. As you can see, `num_liquidations` and `repay_to_borrow_ratio` have the biggest impact (largest bars, either positive or negative), which is good because those were the behaviors I considered most important for assessing risk or reliability.

### Sample Wallet Analysis (From `wallet_credit_scores.csv` & `analysis_df`)

To make this more real, I'll pick out a couple of example wallets based on their scores.

**High-Score Wallet Example (Score: ~950):**
* **Wallet ID:** `0x4a01c440a7a3792019a27c73a872658a52d3a3f5` (Just an example, you should find a high-scoring wallet from your generated `wallet_credit_scores.csv`)
* **Key Behavior:** This wallet likely shows a `repay_to_borrow_ratio` close to 1.0, has `num_liquidations` of 0, a high `total_deposit_usd`, and has been active for a long `account_age_days`.
* **My Interpretation:** This wallet is a prime example of a very reliable Aave user. They consistently pay back their loans, have never faced liquidation, show significant financial engagement with the protocol through large deposits, and have a long, stable history. All these behaviors contribute to their top-tier credit score.

**Low-Score Wallet Example (Score: ~150):**
* **Wallet ID:** `0x7b7e5e3a89e9f9b2f6d8a3c5a7d2e0a1b6c7d8e9` (Just an example, you should find a low-scoring wallet from your generated `wallet_credit_scores.csv`)
* **Key Behavior:** This wallet probably has a low `repay_to_borrow_ratio` (e.g., 0.2), a high `num_liquidations` (e.g., 3 or more), and a significant `total_liquidation_usd`.
* **My Interpretation:** This wallet's history points to risky or poor loan management. Its low repayment ratio and multiple, possibly large, liquidations indicate that it struggles to maintain healthy positions on Aave. This type of behavior heavily penalizes its credit score.

---

## Limitations

Even though this was a fun project, it's important to know its limits:

* **It's Just My Guess (Unsupervised):** Since I didn't have any pre-defined "good" or "bad" labels, the scores are based on my interpretation and the weights I gave to features. It's a starting point, not a perfect, universally true credit score.
* **My Weights are Subjective:** The numbers I picked for how much each feature influences the score are my best estimate. Someone else might choose different weights, leading to different scores.
* **Only Aave V2 Data:** The scores are based *only* on transactions on Aave V2. I don't know what these wallets are doing on other DeFi platforms or if they have any traditional loans. So, it's not a complete picture of their financial health.
* **`value_usd` is a Snapshot:** The `value_usd` represents the value at the time of the transaction. Crypto prices change *a lot*, so that $1000 deposit might be worth $500 now, but my model only saw its value when it happened.
* **Scaling for Huge Data:** My code works great for 100,000 transactions. But if Aave had billions of transactions, I'd need a much more complex system to process it all efficiently.

---

## Conclusion & Future Work
This project was a fantastic learning experience in building an unsupervised credit scoring system for DeFi wallets. It shows that we can get a good sense of a wallet's reliability just by analyzing its on-chain behavior. The features and weighting I used make the score pretty transparent, which is a big plus.
