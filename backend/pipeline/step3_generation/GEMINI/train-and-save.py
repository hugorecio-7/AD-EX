
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import math

from ad_performance_predictor import engineer_features

sns.set_theme(style="whitegrid")

def recursive_autoregressive_predict(model, test_df, target_col, features):
    """
    Performs true recursive forecasting. 
    Predicts day T, then uses that prediction to update lag/rolling features for day T+1.
    """
    print(f"--- Running True Recursive Inference for {target_col} ---")
    
    # Create a copy so we don't overwrite the original actuals just in case
    df_pred = test_df.copy()
    df_pred[f'predicted_{target_col}'] = np.nan
    
    # We must forecast each unique entity (creative + country + OS) independently
    group_cols = ['creative_id', 'country', 'os']
    grouped = df_pred.groupby(group_cols)
    
    processed_groups = []
    
    for name, group in grouped:
        # Ensure we are stepping through time chronologically
        group = group.sort_values('days_since_launch').reset_index(drop=True)
        predictions = []
        
        for i in range(len(group)):
            # 1. Update the current row's lag features using PAST PREDICTIONS
            if i > 0:
                group.loc[i, f'{target_col}_lag_1'] = predictions[i-1]
            if i > 1:
                group.loc[i, f'{target_col}_lag_2'] = predictions[i-2]
                
            # 2. Update the rolling windows using PAST PREDICTIONS (Max 3 days history)
            history = predictions[max(0, i-3):i]
            
            if len(history) > 0:
                group.loc[i, f'{target_col}_rolling_mean_3d'] = np.mean(history)
                if len(history) > 1:
                    # ddof=1 for sample standard deviation
                    group.loc[i, f'{target_col}_rolling_std_3d'] = np.std(history, ddof=1)
                else:
                    group.loc[i, f'{target_col}_rolling_std_3d'] = 0.0
                    
            # 3. Extract the updated feature vector for the current day
            X_row = group.loc[[i], features]
            
            # 4. Predict the current day and store it
            pred = model.predict(X_row)[0]
            predictions.append(pred)
            
        # Assign the recursive predictions back to the group
        group[f'predicted_{target_col}'] = predictions
        processed_groups.append(group)
        
    # Recombine all the independently forecasted series back into one dataframe
    return pd.concat(processed_groups, ignore_index=True)

def train_evaluation_model(df, target_col='CTR', num_test_creatives=6):
    """
    Trains the model once and returns the test dataframe with predictions.
    """
    print(f"--- Training Autoregressive Model for {target_col} ---")
    
    # 1. Isolate test set (Hold out N random creatives entirely)
    unique_creatives = df['creative_id'].unique()
    test_creatives = np.random.choice(unique_creatives, size=num_test_creatives, replace=False)
    
    train_df = df[~df['creative_id'].isin(test_creatives)].copy()
    test_df = df[df['creative_id'].isin(test_creatives)].copy()
    
    print(f"Training on {len(unique_creatives) - num_test_creatives} creatives...")
    print(f"Testing on {len(test_creatives)} held-out creatives...")
    
    # 2. Define features (Ensuring targets are excluded)
    all_targets = ['CTR', 'CVR', 'VTR', 'viewability_rate']
    features = [col for col in df.columns if col not in all_targets and col != 'creative_id']
    
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    
    # 3. Train LightGBM model
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=150)
    
    # 4. Generate predictions for the entire test set
    test_df = recursive_autoregressive_predict(model, test_df, target_col, features)    
    return model, features, test_df, test_creatives

def plot_multiple_creatives(test_df, target_col='CTR', num_cols=2):
    """
    Takes the evaluated test set and plots a grid of multiple creatives.
    """
    # Get the unique creatives in the test set
    creatives_to_plot = test_df['creative_id'].unique()
    num_plots = len(creatives_to_plot)
    num_rows = math.ceil(num_plots / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, creative_id in enumerate(creatives_to_plot):
        ax = axes[idx]
        
        # Filter data for the specific creative
        plot_data = test_df[test_df['creative_id'] == creative_id].copy()
        
        # Keep time-series clean by focusing on its top Country/OS segment
        top_country = plot_data['country'].mode()[0]
        top_os = plot_data['os'].mode()[0]
        plot_data = plot_data[(plot_data['country'] == top_country) & (plot_data['os'] == top_os)]
        plot_data = plot_data.sort_values('days_since_launch')
        
        # Plot Actual vs Predicted
        ax.plot(plot_data['days_since_launch'], plot_data[target_col], marker='o', 
                label='Actual', color='blue', linewidth=2)
        ax.plot(plot_data['days_since_launch'], plot_data['predicted_' + target_col], marker='x', 
                linestyle='--', label='Predicted', color='orange', linewidth=2)
        
        ax.set_title(f'Creative: {creative_id}\nSegment: {top_country} - {top_os}', fontsize=12)
        ax.set_xlabel('Days Since Launch')
        ax.set_ylabel(target_col)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots if the grid isn't perfectly filled
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

# --- Execution Workflow ---

# --- 1. Load Data (Replace with your actual file paths) ---
creatives_df = pd.read_csv('../../frontend/public/data/creatives.csv')
stats_df = pd.read_csv('../../frontend/public/data/creative_daily_country_os_stats.csv')

# --- 2. Call the Feature Engineering Function ---
final_df = engineer_features(creatives_df, stats_df)

# 1. Train the model ONCE and get your test dataframe (e.g., holding out 6 creatives)
model, features, evaluated_test_df, held_out_ids = train_evaluation_model(final_df, target_col='CTR', num_test_creatives=6)


import joblib

# 1. Save the LightGBM model natively
# This creates a lightweight text file containing the tree structures
model.save_model('lgbm_ctr_model.txt')

# 2. Save the feature list 
# The simulator needs this to build the columns in the exact right order
joblib.dump(features, 'lgbm_features.pkl')

# 3. Save the evaluated DataFrame
# Using to_pickle instead of to_csv to preserve the categorical dtypes!
evaluated_test_df.to_pickle('evaluated_test_df.pkl')

print("Model, features, and DataFrames successfully saved!")

# # 2. Plot all 6 held-out creatives in a beautiful 2-column grid
# plot_multiple_creatives(evaluated_test_df, target_col='CTR', num_cols=2)

# # (Optional) Look at feature importance later without retraining
# lgb.plot_importance(model, max_num_features=15, importance_type='gain')
# plt.show()