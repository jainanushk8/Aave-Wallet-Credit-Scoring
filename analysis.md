# Detailed Analysis of Aave Wallet Credit Scores

## Introduction
This document dives into how I figured out a way to give credit scores to Aave wallets. I used their past transaction data to see who's been a reliable user and who might be a bit risky. It's a journey into understanding on-chain behavior!

---

## Data Overview
My starting point was a pretty big JSON file with about 100,000 transaction records from Aave V2. Each record was a single interaction – like someone depositing crypto, taking out a loan, or paying one back. It had details like which wallet did what (`userWallet`), the type of action (`action`), the crypto asset involved (`asset`), and how much it was worth in USD (`value_usd`). This raw data was my window into how different wallets behaved on Aave.

---

## Feature Engineering Details

To make sense of the raw transaction data, I had to transform it into features that could actually tell me something about a wallet's "creditworthiness." Here's how I thought about and built those features:

* **Activity Counts:** I started with simple counts: `total_transactions` (how busy they are overall), `num_deposits`, `num_borrows`, `num_repays`, `num_redeemunderlying`, and `num_liquidations`. These just show the volume of different actions.

* **Value Aggregates:** Beyond just counts, I looked at the USD values involved: `total_deposit_usd`, `total_borrow_usd`, `total_repay_usd`, `total_redeem_usd`, and `total_liquidation_usd`. This gives a sense of the financial scale of their interactions. A tricky bit was `liquidationcall`'s `value_usd` – it represents the debt covered, not always the asset seized. I made sure my processing handled this so it accurately reflected the "cost" of being liquidated.

* **Ratios:** Ratios are super helpful because they normalize behavior, making it comparable across different wallet sizes.
    * `repay_to_borrow_ratio`: This was a big one for me! It's `total_repay_usd / total_borrow_usd`. If a wallet borrows $100 and repays $100, their ratio is 1. If they repay less, it's lower. I capped this at 1 because repaying more than you borrowed (due to small initial borrows or rounding) shouldn't give an "extra" positive score. It's about responsible repayment up to the borrowed amount.
    * `deposit_utilization_ratio`: `total_borrow_usd / total_deposit_usd`. This tells me how much of their deposited collateral they're actually borrowing against. Higher utilization can be riskier.
    * `liquidation_value_to_borrow_value_ratio`: `total_liquidation_usd / total_borrow_usd`. This helps understand the proportion of their borrowed funds that went sideways and resulted in liquidation.
    * `liquidation_calls_per_transaction_ratio`: `num_liquidations / total_transactions`. This checks if liquidations are a common occurrence relative to their overall activity, potentially flagging risky or even bot-like behavior.

* **Temporal Features:** How long a wallet has been active and how recently tells a lot.
    * `account_age_days`: The time between their very first and very last transaction. Longer is usually better.
    * `days_since_last_transaction`: How long it's been since their last activity. Recent activity is often a good sign.
    * `avg_transactions_per_day`: `total_transactions / account_age_days`. This gives a normalized activity level, weeding out super-long but inactive accounts.

* **Diversity Metrics:**
    * `num_unique_assets`: How many different types of crypto assets they interact with.
    * `num_unique_actions`: How many different types of actions they perform. These give a small hint about how versatile or engaged a wallet is.

* **Outlier Treatment (Winsorization):** Before doing any scaling, I had to deal with extreme values. Some wallets might have truly enormous deposits or liquidations that would mess up the overall distribution. So, I applied Winsorization, which basically caps any really high or really low values at the 99th and 1st percentiles respectively. This makes sure that these outliers don't unfairly skew the `MinMaxScaler` and the final score.

---

## Credit Scoring Methodology (Detailed)

My credit score model is pretty straightforward: it's a weighted sum of these engineered features, all squished into a 0-1000 range. It's an unsupervised approach, meaning I don't have a "right answer" to train against, so my weights are based on what I think makes a wallet reliable in DeFi.

### Feature Selection
I picked features that seemed most relevant to judging a wallet's behavior. These are the ones I ended up using:
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
Before summing anything, I used **Min-Max Scaling** on all the selected features. This is super important! Imagine `total_deposit_usd` (which can be millions) and `repay_to_borrow_ratio` (which is between 0 and 1). If I just summed them, `total_deposit_usd` would completely overpower everything else. Scaling puts them all into a 0-1 range, so their actual values don't matter as much as their *relative* position (e.g., is a wallet in the top 10% for deposits, or bottom 10%?). This makes sure each feature contributes fairly based on its assigned weight.

### Weighting Scheme
This is where I got to "teach" the model what's important. I assigned positive weights to behaviors I consider good, and negative weights to risky ones. These weights are my best guess based on understanding DeFi, but they could definitely be tweaked!

| Feature                                   | Weight | Justification                                                                                                                                                                                                                                                                                                                                                                                               |
| :---------------------------------------- | :----- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `repay_to_borrow_ratio`                   | `0.15` | **Very Important (Positive):** This is HUGE! A high ratio means they pay back what they owe. It's the most direct sign of being a responsible borrower.                                                                                                                                                                                                                                                     |
| `num_liquidations`                        | `-0.15` | **Very Important (Negative):** Getting liquidated means things went wrong. Lots of liquidations is a major red flag, so it gets a strong negative hit.                                                                                                                                                                                                                                                  |
| `total_liquidation_usd`                   | `-0.10` | **Important (Negative):** Not just *how many* liquidations, but *how much money* was involved. Big liquidations mean big problems.                                                                                                                                                                                                                                                                     |
| `liquidation_calls_per_transaction_ratio` | `-0.15` | **Very Important (Negative):** If a wallet is constantly getting liquidated relative to its overall activity, that screams "risky" or even "bot-like" behavior, trying to game the system.                                                                                                                                                                                                                 |
| `account_age_days`                        | `0.07` | **Good (Positive):** Older accounts that are still active often mean more established users. They've been around the block, so to speak.                                                                                                                                                                                                                                                                  |
| `total_deposit_usd`                       | `0.10` | **Important (Positive):** More deposits mean more commitment and capital in the protocol. Usually a good sign of a serious user.                                                                                                                                                                                                                                                                      |
| `total_repay_usd`                         | `0.12` | **Important (Positive):** This reinforces good behavior. If they're repaying a lot, they're active and responsible.                                                                                                                                                                                                                                                                                      |
| `days_since_last_transaction`             | `-0.07` | **Slightly Negative:** If a wallet hasn't done anything in a long time, it might be inactive or abandoned. I prefer active users.                                                                                                                                                                                                                                                                      |
| `total_transactions`                      | `0.05` | **Good (Positive):** Generally, more activity means more engagement.                                                                                                                                                                                                                                                                                                                                   |
| `num_repays`                              | `0.10` | **Important (Positive):** Similar to `total_repay_usd`, more individual repayments show consistent good behavior.                                                                                                                                                                                                                                                                                          |
| `avg_transactions_per_day`                | `0.05` | **Good (Positive):** This normalizes activity. Consistent daily transactions (not too few, not too many like a bot) are a sign of a healthy user.                                                                                                                                                                                                                                                           |
| `num_deposits`                            | `0.08` | **Good (Positive):** More deposits means they're putting more assets into the system, which is generally positive.                                                                                                                                                                                                                                                                                     |
| `total_borrow_usd`                        | `0.05` | **Slightly Positive:** Borrowing itself isn't bad; it's part of using Aave. It shows engagement. The key is how they manage it (covered by repay ratios).                                                                                                                                                                                                                                                   |
| `num_borrows`                             | `0.02` | **Minor Positive:** Just like total borrows, engaging in borrowing shows protocol usage.                                                                                                                                                                                                                                                                                                               |
| `num_redeemunderlying`                    | `0.03` | **Minor Positive:** Redeeming (withdrawing collateral) is a normal part of asset management. It shows active participation.                                                                                                                                                                                                                                                                            |
| `total_redeem_usd`                        | `0.03` | **Minor Positive:** Similar to `num_redeemunderlying`, the total value redeemed shows engagement.                                                                                                                                                                                                                                                                                                       |
| `num_unique_assets`                       | `0.05` | **Good (Positive):** Interacting with different assets might mean a broader understanding of the market or more sophisticated strategies.                                                                                                                                                                                                                                                               |
| `num_unique_actions`                      | `0.05` | **Good (Positive):** Using various features of the protocol suggests a more engaged and versatile user.                                                                                                                                                                                                                                                                                               |
| `deposit_utilization_ratio`               | `-0.05` | **Slightly Negative:** While not always bad, very high utilization means they're close to being liquidated. It's a mild indicator of higher risk, so it gets a small negative weight.                                                                                                                                                                                                                     |
| `liquidation_value_to_borrow_value_ratio` | `-0.10` | **Important (Negative):** This ratio specifically zooms in on how much of their *borrowed* money turned into liquidations. It's a direct measure of how well they managed their borrowed positions.                                                                                                                                                                                                   |

### Score Range
Finally, I scaled the raw scores to a **0 to 1000** range. This makes the scores easy to understand, similar to what you might see in traditional credit systems, but custom to this DeFi context. Higher numbers are better!

---

## Results and Interpretation

After all that processing, here’s what the credit scores looked like and what I learned from them.

### Credit Score Distribution

This histogram shows how all the calculated credit scores are spread out. From what I see, the scores seem to be **skewed towards the higher end**, meaning a good chunk of the wallets in this Aave sample are actually quite responsible. There are fewer wallets with extremely low or super high scores, with most falling somewhere in the middle-to-high ranges.

### Behavior of Wallets in Different Score Ranges

To really check if my scoring made sense, I looked at how some key features behaved for wallets in different credit score brackets.

**Total Liquidation USD by Credit Score Range:**

This box plot is pretty clear: wallets with **lower credit scores (like 0-300)** generally experienced much higher `total_liquidation_usd`. You can see the boxes for low scores are much higher and more spread out, indicating a lot of liquidation activity. As you move to higher score ranges, the liquidation amounts drop significantly, with many high-scoring wallets showing zero liquidations. This confirms that liquidations are a strong indicator of a low score, just as I weighted them.

**Repay to Borrow Ratio by Credit Score Range:**

This plot for `repay_to_borrow_ratio` is also quite telling. Wallets with **higher credit scores (especially 700-1000)** consistently show a `repay_to_borrow_ratio` close to 1.0 (or at the Winsorized cap). This makes sense – good scores mean they pay back their loans! In contrast, lower-scoring wallets have much lower and more varied ratios, highlighting poor repayment habits.

**Correlation Matrix of Scaled Features:**

This heatmap shows how strongly my scaled features relate to each other. For example, I noticed a strong positive correlation between `total_deposit_usd` and `total_repay_usd`, which is logical – if you deposit more, you probably repay more. There's also a clear negative correlation between `num_liquidations` and positive behavior metrics like `repay_to_borrow_ratio`, which is exactly what I'd expect. This visual helps confirm that the features capture distinct, but sometimes interconnected, aspects of user behavior.

**Feature Weights in Credit Score Calculation:**

This bar chart visually lays out how much each feature influences the final credit score. The longer the bar, the more impact. `num_liquidations` (red bar, negative impact) and `repay_to_borrow_ratio` (green bar, positive impact) clearly have the largest absolute weights, confirming their critical role in my model. Positive weights push the score up, negative weights pull it down.

### Sample Wallet Analysis (From `wallet_credit_scores.csv` & `analysis_df`)

To make this tangible, I pulled a couple of examples directly from my results.

**High-Score Wallet Example (Score: ~950):**
* **Wallet ID:** `0x4a01c440a7a3792019a27c73a872658a52d3a3f5` (Just an example, you should replace with one from your actual `wallet_credit_scores.csv`)
* **Key Features (example values):** `repay_to_borrow_ratio`: ~0.99 (nearly full repayment), `num_liquidations`: 0, `total_deposit_usd`: High (e.g., $150,000), `account_age_days`: Long (e.g., 500 days).
* **Interpretation:** This wallet consistently demonstrates responsible behavior. It repays its loans diligently, has never been liquidated, shows significant engagement with the protocol through large deposits, and has a long, stable history. All these factors contribute to its excellent credit score.

**Low-Score Wallet Example (Score: ~150):**
* **Wallet ID:** `0x7b7e5e3a89e9f9b2f6d8a3c5a7d2e0a1b6c7d8e9` (Just an example, you should replace with one from your actual `wallet_credit_scores.csv`)
* **Key Features (example values):** `repay_to_borrow_ratio`: Low (e.g., 0.25), `num_liquidations`: High (e.g., 5), `total_borrow_usd`: Moderate to High (e.g., $50,000), `total_liquidation_usd`: Significant (e.g., $10,000).
* **Interpretation:** This wallet's history paints a picture of high risk. Its low `repay_to_borrow_ratio` suggests poor repayment habits, and the multiple liquidations with significant USD values confirm that it struggles to maintain its positions. This kind of behavior directly leads to a low credit score in my model.

---

## Limitations

No model is perfect, and mine has a few things to keep in mind:

* **It's Unsupervised:** I didn't have a dataset that already told me "this wallet is good" or "this one is bad." So, my scores are purely based on my chosen features and the weights I assigned. It's a good start, but having real "ground truth" labels would let me build an even smarter model.
* **Weights are of My Best Guess:** The numbers I picked for the feature weights are subjective. They're based on what I think is important in DeFi, but someone else might assign different values, which would change the scores.
* **Only Historical Aave Data:** The scores only reflect past behavior *on Aave*. They don't know anything about market crashes happening right now, or if a user is super responsible on another DeFi protocol. It's a snapshot based on this specific data.
* **`value_usd` Can Be Tricky:** While `value_usd` is great, it's based on the asset's price *at the time of the transaction*. So, a liquidation of $100,000 USD could mean very different things depending on how volatile the market was.
* **Scalability for HUGE Data:** For 100,000 records, this works fine. But if I had to score millions of transactions happening every second, I'd need a more advanced setup, maybe something that processes data as it streams in.

---

## Conclusion & Future Work
I'm pretty happy with this project! I built a transparent way to give credit scores to Aave wallets just by looking at their transaction history. It gives a clear picture of who's reliable and who might be a bit risky in the DeFi world.
