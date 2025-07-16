# Aave Wallet Credit Scoring

This project is all about trying to figure out how "creditworthy" different wallets are on the Aave V2 decentralized finance (DeFi) protocol. Since there's no traditional credit score in DeFi, I built a system to give wallets a score based on their past transaction history. 

---

## The Big Idea (Method Chosen)

My main goal was to come up with a score for Aave wallets from scratch, without any pre-labeled "good" or "bad" examples. This is called an **unsupervised learning** approach. I decided to go with a **weighted linear combination** method. This basically means I picked a bunch of important behaviors (like paying back loans or getting liquidated), gave them scores (weights), and then added them all up. Wallets doing "good" stuff get a boost, and "bad" stuff pulls their score down. Simple, right?

---

## How It All Works (Complete Architecture & Processing Flow)

Here's a breakdown of the steps I took to get from raw Aave data to a credit score:

1.  **Data Collection:**
    * First, I got a big JSON file (`user-wallet-transactions.json`) with tons of past Aave V2 transactions. This file has details like who did what, with which crypto, and how much it was worth in USD.
    * **Data Source:** This data was pulled directly from the Aave V2 protocol, specifically focusing on supply, borrow, repay, and liquidation events.

2.  **Data Loading & Initial Cleanup:**
    * I loaded this JSON data into a pandas DataFrame.
    * The initial step was to just get it into a format I could work with. I also made sure to handle any missing values in the `value_usd` column, especially for liquidation transactions, by making smart guesses based on the `amount` and `asset` fields if needed.

3.  **Feature Engineering:**
    * This was a big part! I created new pieces of information (features) from the raw transaction data that I thought would be useful for judging a wallet.
    * **Activity Counts:** I counted things like how many times a wallet deposited, borrowed, or got liquidated (`num_deposits`, `num_borrows`, `num_liquidations`, etc.).
    * **Value Aggregates:** I summed up the total USD value for each type of transaction (`total_deposit_usd`, `total_borrow_usd`, `total_liquidation_usd`, etc.).
    * **Ratios:** I calculated important ratios like `repay_to_borrow_ratio` (did they repay what they borrowed?) and `liquidation_value_to_borrow_value_ratio` (how much borrowed value got liquidated?). These are key for showing responsible behavior.
    * **Temporal Features:** I looked at how old an account was (`account_age_days`) and how recently they last transacted (`days_since_last_transaction`).
    * **Diversity Metrics:** How many different assets or actions a wallet used.

4.  **Outlier Treatment (Winsorization):**
    * Some wallets had crazy high values for certain features (like one huge deposit). These extreme values (outliers) could mess up my scoring. So, I used something called **Winsorization** to "cap" these values at the 1st and 99th percentiles. It means really high values get pulled down to the 99th percentile, and really low ones get pulled up to the 1st percentile. This makes the data much more well-behaved for the next step.

5.  **Feature Standardization (Min-Max Scaling):**
    * Imagine if `total_deposit_usd` was in the millions and `repay_to_borrow_ratio` was between 0 and 1. If I just added them up, deposits would completely dominate the score! So, I used **Min-Max Scaling** to transform every feature's values to a range between 0 and 1. This ensures that every feature, no matter its original scale, contributes fairly according to its assigned weight.

6.  **Credit Score Calculation:**
    * This is the core! I assigned specific **weights** to each of the scaled features. For example, a high `repay_to_borrow_ratio` gets a positive weight (good for score), while a high `num_liquidations` gets a negative weight (bad for score).
    * I multiplied each scaled feature by its weight and summed them all up for each wallet.
    * The raw scores were then scaled again to fit a nice **0 to 1000** range, making them easy to interpret (higher is better!).

7.  **Output:**
    * Finally, the scores are saved into a CSV file (`wallet_credit_scores.csv`), showing each wallet and its calculated score.

This whole process is encapsulated in the `generate_credit_scores.py` script.

---

## How to Use This Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/Aave-Wallet-Credit-Scoring.git](https://github.com/your-username/Aave-Wallet-Credit-Scoring.git)
    cd Aave-Wallet-Credit-Scoring
    ```
    (Replace `your-username` with your GitHub username.)

2.  **Get the Data:**
    * The `user-wallet-transactions.json` file is quite large. For this project, I've stored it in the same directory. If it's too big for direct GitHub hosting, I'd provide a Google Drive link to download it here. Make sure this file is in the root of your project directory.

3.  **Install Dependencies:**
    * You'll need Python installed. Then, create a virtual environment (good practice!) and install the necessary libraries:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

4.  **Run the Scoring Script:**
    ```bash
    python generate_credit_scores.py
    ```
    This script will:
    * Load the transaction data.
    * Engineer features.
    * Calculate credit scores.
    * Save the `wallet_credit_scores.csv` file.
    * Generate the visualization images (`.png` files) that are used in `analysis.md`.

5.  **Explore the Analysis:**
    * Open `analysis.md` in a Markdown viewer (like on GitHub itself or VS Code) to see a detailed explanation of the findings, including graphs.

---

## Project Structure
.
├── generate_credit_scores.py  # The main script to process data and calculate scores
├── user-wallet-transactions.json # Raw Aave V2 transaction data (large file)
├── wallet_credit_scores.csv    # Output: CSV with wallet IDs and their credit scores
├── README.md                   # You are reading it!
├── analysis.md                 # Detailed findings, graphs, and interpretations
├── credit_score_distribution.png
├── liquidation_usd_by_score_range.png
├── repay_borrow_ratio_by_score_range.png
├── features_correlation_heatmap.png
└── feature_weights_chart.png

---

## What I Learned
This project was a great way to learn about:
* Handling real-world, messy data (especially dealing with `value_usd` and potential outliers).
* The importance of **feature engineering** to turn raw data into meaningful insights.
* Applying **data preprocessing** techniques like Winsorization and Min-Max Scaling.
* Building a simple yet effective **unsupervised credit scoring model**.
* **Visualizing data** to understand distributions and relationships.

It was challenging but super rewarding to see how different on-chain behaviors actually translate into a "score"!
