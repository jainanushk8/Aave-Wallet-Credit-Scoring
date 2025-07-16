
import pandas as pd
import numpy as np
import json
import zipfile
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress all warnings for cleaner output in this consolidated script
warnings.filterwarnings('ignore')

def generate_credit_scores(json_file_path='user-wallet-transactions.json', output_csv_path='wallet_credit_scores.csv'):
    """
    Generates credit scores for Aave wallet users from raw JSON data.

    Args:
        json_file_path (str): Path to the input JSON data file.
        output_csv_path (str): Path to save the output CSV file with wallet credit scores.
    """
    print(f"Starting credit score generation process using '{json_file_path}'...")

    # --- Phase 1: Data Loading & Initial Inspection ---
    print("Phase 1: Loading and initially inspecting data...")
    try:
        if json_file_path.endswith('.zip'):
            # Assume it's a zip file containing a JSON. Extract it.
            print(f"Extracting '{json_file_path}'...")
            with zipfile.ZipFile(json_file_path, 'r') as zip_ref:
                # Assuming the JSON file inside the zip has the same name but with .json extension
                # Or you might need to list zip_ref.namelist() to find the correct file
                extracted_json_name = os.path.basename(json_file_path).replace('.zip', '.json')
                zip_ref.extract(extracted_json_name, path='.')
                temp_json_path = extracted_json_name
                print(f"Extracted to '{temp_json_path}'")
        else:
            temp_json_path = json_file_path

        # This part assumes a file with one JSON object per line.
        # If the file is a single JSON array of objects, use json.load(f)
        # Based on previous outputs, it seems to be a list of JSON objects directly,
        # so json.load(f) is appropriate if the file contains a single JSON array.
        # If it's one JSON object per line, list comprehension with json.loads(line) is correct.
        # Reverting to json.load(f) as it was used in previous blocks for clarity on structure.
        with open(temp_json_path, 'r') as f:
            data = json.load(f) # Assuming the entire file is a single JSON array
        print(f"Successfully loaded {len(data)} records from '{temp_json_path}'.")

    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}' or extracted file '{temp_json_path}'.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{temp_json_path}'. Check file format. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    df = pd.DataFrame(data)

    # --- Phase 2: Feature Engineering ---
    print("\nPhase 2: Performing Feature Engineering...") # Escaped newline for the script's output

    # Convert timestamp to datetime objects
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')

    # Handle 'amount' and 'assetPriceUSD' which can be strings or nested for liquidationcall
    def get_value_usd(row):
        try:
            action_data = row['actionData']
            action = row['action']

            # Specific handling for 'liquidationcall'
            if action == 'liquidationcall' and isinstance(action_data, dict):
                collateral_amount_str = action_data.get('collateralAmount', '0')
                collateral_price_str = action_data.get('collateralAssetPriceUSD', '0')
                collateral_symbol = action_data.get('collateralReserveSymbol', '') # Added symbol for decimals lookup

                if collateral_amount_str and collateral_price_str and float(collateral_price_str) > 0:
                    amount = float(collateral_amount_str)
                    price = float(collateral_price_str)
                    # For WETH, decimals are 18. Adjust based on symbol.
                    decimals = {
                        'USDC': 6, 'USDT': 6, 'DAI': 18, 'WETH': 18, 'WMATIC': 18,
                        'LINK': 18, 'AAVE': 18, 'WBTC': 8
                    }.get(collateral_symbol, 18) # Default to 18 if symbol not found
                    return (amount / (10**decimals)) * price
                else:
                    return 0.0 # Could not extract for liquidator

            # General handling for other actions ('deposit', 'borrow', 'repay', 'redeemUnderlying')
            if isinstance(action_data, dict):
                amount_str = action_data.get('amount', '0')
                price_str = action_data.get('assetPriceUSD', '0')
                asset_symbol = action_data.get('assetSymbol', '')
            else: # Fallback if actionData is not a dict
                amount_str = row.get('amount', '0')
                price_str = row.get('assetPriceUSD', '0')
                asset_symbol = row.get('assetSymbol', '')

            if not amount_str or not price_str:
                return 0.0

            amount = float(amount_str)
            price = float(price_str)

            # Define common asset decimals within the function for self-containment
            asset_decimals_map = {
                'USDC': 6, 'USDT': 6, 'DAI': 18, 'WETH': 18, 'WMATIC': 18,
                'LINK': 18, 'AAVE': 18, 'WBTC': 8, 'ETH': 18
            }
            # Get decimals, default to 18
            decimals = asset_decimals_map.get(asset_symbol, 18)

            # Scale amount by its decimals
            scaled_amount = amount / (10**decimals)
            return scaled_amount * price

        except (ValueError, TypeError) as e:
            # print(f"Warning: Data conversion error for row {row.get('_id', '')}: {e}") # For debugging
            return 0.0
        except Exception as e:
            # print(f"An unexpected error in get_value_usd for row {row.get('_id', '')}: {e}") # For debugging
            return 0.0

    df['value_usd'] = df.apply(get_value_usd, axis=1)

    # Drop rows where essential calculations resulted in NaNs in value_usd
    df.dropna(subset=['value_usd', 'userWallet', 'action', 'timestamp_dt'], inplace=True)


    # Get the latest transaction timestamp in the dataset to calculate recency relative to the end of the data.
    current_date = df['timestamp_dt'].max() # Or pd.Timestamp.now(tz='UTC') for current time

    # Group by userWallet to create wallet-level features
    print("Aggregating features by user wallet...")
    wallet_features = df.groupby('userWallet').agg(
        total_transactions=('txHash', 'nunique'),
        num_deposits=('action', lambda x: (x == 'deposit').sum()),
        num_borrows=('action', lambda x: (x == 'borrow').sum()),
        num_repays=('action', lambda x: (x == 'repay').sum()),
        num_redeemunderlying=('action', lambda x: (x == 'redeemunderlying').sum()), # Corrected action name
        num_liquidations=('action', lambda x: (x == 'liquidationcall').sum()),
        total_deposit_usd=('value_usd', lambda x: x[df.loc[x.index, 'action'] == 'deposit'].sum()),
        total_borrow_usd=('value_usd', lambda x: x[df.loc[x.index, 'action'] == 'borrow'].sum()),
        total_repay_usd=('value_usd', lambda x: x[df.loc[x.index, 'action'] == 'repay'].sum()),
        total_redeem_usd=('value_usd', lambda x: x[df.loc[x.index, 'action'] == 'redeemunderlying'].sum()), # Corrected action name
        total_liquidation_usd=('value_usd', lambda x: x[df.loc[x.index, 'action'] == 'liquidationcall'].sum()),
        # Corrected method to get unique asset symbols, handling potential None/NaN
        num_unique_assets=('actionData', lambda x: x.apply(lambda ad: ad.get('assetSymbol') if isinstance(ad, dict) else None).dropna().nunique()),
        first_transaction_date=('timestamp_dt', 'min'),
        last_transaction_date=('timestamp_dt', 'max'),
        num_unique_actions=('action', 'nunique')
    ).reset_index()

    # Calculate account_age_days
    wallet_features['account_age_days'] = (current_date - wallet_features['first_transaction_date']).dt.days

    # Calculate days since last transaction
    wallet_features['days_since_last_transaction'] = (current_date - wallet_features['last_transaction_date']).dt.days

    # Fill NaN values in sum-based features with 0 (if no transactions of that type occurred)
    sum_cols = ['total_deposit_usd', 'total_borrow_usd', 'total_repay_usd',
                'total_redeem_usd', 'total_liquidation_usd']
    for col in sum_cols:
        wallet_features[col] = wallet_features[col].fillna(0)

    # Feature ratios and more complex metrics
    epsilon = 1e-9 # Small value to prevent division by zero

    wallet_features['repay_to_borrow_ratio'] = wallet_features.apply(
        lambda row: row['total_repay_usd'] / (row['total_borrow_usd'] + epsilon) if row['total_borrow_usd'] >= 0 else 0, # >= 0 to handle potential negative values (unlikely here)
        axis=1
    )
    wallet_features['repay_to_borrow_ratio'] = wallet_features['repay_to_borrow_ratio'].replace([np.inf, -np.inf], 0).clip(upper=1) # Cap at 1

    wallet_features['liquidation_value_to_borrow_value_ratio'] = wallet_features.apply(
        lambda row: row['total_liquidation_usd'] / (row['total_borrow_usd'] + epsilon) if row['total_borrow_usd'] >= 0 else 0,
        axis=1
    )
    wallet_features['liquidation_value_to_borrow_value_ratio'] = wallet_features['liquidation_value_to_borrow_value_ratio'].replace([np.inf, -np.inf], 0)

    wallet_features['liquidation_calls_per_transaction_ratio'] = wallet_features.apply(
        lambda row: row['num_liquidations'] / (row['total_transactions'] + epsilon) if row['total_transactions'] >= 0 else 0,
        axis=1
    )
    wallet_features['liquidation_calls_per_transaction_ratio'] = wallet_features['liquidation_calls_per_transaction_ratio'].replace([np.inf, -np.inf], 0)

    wallet_features['avg_transactions_per_day'] = wallet_features.apply(
        lambda row: row['total_transactions'] / (row['account_age_days'] + epsilon) if row['account_age_days'] >= 0 else 0,
        axis=1
    )
    wallet_features['avg_transactions_per_day'] = wallet_features['avg_transactions_per_day'].replace([np.inf, -np.inf], 0)

    wallet_features['deposit_utilization_ratio'] = wallet_features.apply(
        lambda row: row['total_borrow_usd'] / (row['total_deposit_usd'] + epsilon) if row['total_deposit_usd'] >= 0 else 0,
        axis=1
    )
    wallet_features['deposit_utilization_ratio'] = wallet_features['deposit_utilization_ratio'].replace([np.inf, -np.inf], 0).clip(upper=1)


    # --- Outlier Treatment (Winsorization) ---
    print("Applying outlier treatment (Winsorization)...")
    skewed_features = [
        'total_transactions', 'num_deposits', 'num_borrows', 'num_repays',
        'num_redeemunderlying', 'total_deposit_usd', 'total_borrow_usd',
        'total_repay_usd', 'total_redeem_usd', 'total_liquidation_usd',
        'repay_to_borrow_ratio', 'liquidation_value_to_borrow_value_ratio',
        'liquidation_calls_per_transaction_ratio', 'deposit_utilization_ratio',
        'avg_transactions_per_day'
    ]

    for col in skewed_features:
        if col in wallet_features.columns and wallet_features[col].dtype != 'object':
            # Cap at 99th percentile (upper bound only for positive values)
            upper_bound = wallet_features[col].quantile(0.99)
            wallet_features[col] = np.where(wallet_features[col] > upper_bound, upper_bound, wallet_features[col])
            # For features like 'days_since_last_transaction', a lower bound might also be useful but 0 is usually fine.
    print("Outlier treatment applied.")

    # Drop intermediate datetime columns used for calculations
    wallet_features = wallet_features.drop(columns=['first_transaction_date', 'last_transaction_date'])
    print("Feature engineering complete.")

    # --- Phase 3: Credit Scoring Model ---
    print("\nPhase 3: Calculating Credit Scores...") # Escaped newline

    scoring_df = wallet_features.copy()

    # Define the features to be used for scoring
    features_for_scoring = [
        'total_transactions', 'num_deposits', 'num_borrows', 'num_repays',
        'num_redeemunderlying', 'num_liquidations', 'total_deposit_usd',
        'total_borrow_usd', 'total_repay_usd', 'total_redeem_usd',
        'total_liquidation_usd', 'num_unique_assets', 'account_age_days',
        'days_since_last_transaction', 'repay_to_borrow_ratio',
        'liquidation_value_to_borrow_value_ratio',
        'liquidation_calls_per_transaction_ratio',
        'deposit_utilization_ratio', 'num_unique_actions',
        'avg_transactions_per_day'
    ]
    features_for_scoring = [f for f in features_for_scoring if f in scoring_df.columns and scoring_df[f].dtype != 'object']

    # Handle potential NaNs/Infs before scaling that might result from edge cases
    for col in features_for_scoring:
        scoring_df[col] = scoring_df[col].replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with NaN then fill NaN with 0

    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    scoring_df[features_for_scoring] = scaler.fit_transform(scoring_df[features_for_scoring])
    print("Features scaled.")

    feature_weights = {
        'total_transactions': 0.05,
        'num_deposits': 0.08,
        'num_borrows': 0.02,
        'num_repays': 0.10,
        'num_redeemunderlying': 0.03,
        'total_deposit_usd': 0.10,
        'total_borrow_usd': 0.05,
        'total_repay_usd': 0.12,
        'total_redeem_usd': 0.03,
        'num_unique_assets': 0.05,
        'account_age_days': 0.07,
        'repay_to_borrow_ratio': 0.15,
        'num_unique_actions': 0.05,
        'avg_transactions_per_day': 0.05,

        'num_liquidations': -0.15,
        'total_liquidation_usd': -0.10,
        'liquidation_value_to_borrow_value_ratio': -0.10,
        'liquidation_calls_per_transaction_ratio': -0.15,
        'days_since_last_transaction': -0.07,
        'deposit_utilization_ratio': -0.05
    }

    print("Calculating raw credit scores...")
    scoring_df['raw_score'] = 0.0

    for feature, weight in feature_weights.items():
        if feature in scoring_df.columns and feature in features_for_scoring: # Ensure it's a numeric feature that was scaled
            if weight >= 0:
                scoring_df['raw_score'] += scoring_df[feature] * weight
            else:
                scoring_df['raw_score'] -= scoring_df[feature] * abs(weight)
        # else:
            # print(f"Warning: Feature '{feature}' from weights not found or not used for scoring.")

    raw_score_min = scoring_df['raw_score'].min()
    raw_score_max = scoring_df['raw_score'].max()

    if (raw_score_max - raw_score_min) == 0:
        scoring_df['normalized_score'] = 0.5
    else:
        scoring_df['normalized_score'] = (scoring_df['raw_score'] - raw_score_min) / (raw_score_max - raw_score_min)

    scoring_df['credit_score'] = scoring_df['normalized_score'] * 1000
    scoring_df['credit_score'] = scoring_df['credit_score'].astype(int)

    print("Credit scores calculated.")

    # Save the scores to a CSV
    scoring_df[['userWallet', 'credit_score']].to_csv(output_csv_path, index=False)
    print(f"Wallet credit scores saved to '{output_csv_path}'")
    print("Credit score generation process completed.")

if __name__ == "__main__":
    generate_credit_scores()
