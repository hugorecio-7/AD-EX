import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

# --- 1. Load the Artifacts ---
print("Loading model and artifacts...")

# Load the model
model = lgb.Booster(model_file='lgbm_ctr_model.txt')

# Load the exact feature list used during training
features = joblib.load('lgbm_features.pkl')

# Load the dataframe (if you want to run the multi-plot visualization)
# evaluated_test_df = pd.read_pickle('evaluated_test_df.pkl')

# --- 2. Paste the simulate_custom_creative function here ---
def simulate_custom_creative(model, creative_params, segment_params, model_features, target_col='CTR', num_days=30):
    """
    Simulates a 30-day campaign for a completely new, custom creative.
    """
    print(f"--- Simulating {num_days}-Day Campaign for Custom Creative ---")
    
    # 1. Create a blank timeline dataframe
    df = pd.DataFrame({'days_since_launch': range(1, num_days + 1)})
    
    # 2. Inject segment and static creative parameters
    # This broadcasts your static values (like 'format' or 'country') across all 30 days
    for key, value in {**segment_params, **creative_params}.items():
        df[key] = value
        
    # 3. Apply the Cold Start Flag
    df['is_cold_start'] = (df['days_since_launch'] <= 3).astype(int)
    
    # 4. Initialize lag and rolling features as NaN
    # LightGBM will use its native missing-value handling for the early days
    lag_cols = [col for col in model_features if 'lag' in col or 'rolling' in col]
    for col in lag_cols:
        df[col] = np.nan
        
    # 5. Ensure categorical columns are properly cast
    categorical_cols = [
        'country', 'os', 'vertical', 'format', 'language', 'theme', 
        'hook_type', 'dominant_color', 'emotional_tone'# , 'cta_text', 
        # 'headline', 'subhead'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Make sure columns are strictly in the order the model expects
    df_features = df[model_features].copy()
            
    # 6. Run the True Recursive Inference
    predictions = []
    
    for i in range(len(df_features)):
        # Update lags dynamically based on the history we just predicted
        if i > 0:
            df_features.loc[i, f'{target_col}_lag_1'] = predictions[i-1]
        if i > 1:
            df_features.loc[i, f'{target_col}_lag_2'] = predictions[i-2]
            
        # Update rolling windows
        history = predictions[max(0, i-3):i]
        if len(history) > 0:
            df_features.loc[i, f'{target_col}_rolling_mean_3d'] = np.mean(history)
            if len(history) > 1:
                df_features.loc[i, f'{target_col}_rolling_std_3d'] = np.std(history, ddof=1)
            else:
                df_features.loc[i, f'{target_col}_rolling_std_3d'] = 0.0
                
        # Extract current day features
        X_row = df_features.loc[[i]]
        
        # Predict and store
        pred = model.predict(X_row)[0]
        predictions.append(pred)
        
    # 7. Attach predictions to our display dataframe
    df[f'predicted_{target_col}'] = predictions
    
    return df
# --- 3. Run the Simulation ---
segment_params = {
    'country': 'US',
    'os': 'iOS'
}

creative_params = {
    'vertical': 'Gaming',
    'format': 'Video',
    'language': 'EN',
    'theme': 'Action',
    'hook_type': 'Gameplay Reveal',
    'dominant_color': 'Red',
    'emotional_tone': 'Exciting',
    'width': 1080,
    'height': 1920,
    'duration_sec': 15,
    'text_density': 0.12,
    'copy_length_chars': 45,
    'readability_score': 8.5,
    'brand_visibility_score': 0.9,
    'clutter_score': 0.3,
    'novelty_score': 0.8,
    'motion_score': 0.95,
    'faces_count': 0,
    'product_count': 1,
    'has_price': 0,
    'has_discount_badge': 0,
    'has_gameplay': 1,
    'has_ugc_style': 0
}

sim_df = simulate_custom_creative(
    model=model, 
    creative_params=creative_params, 
    segment_params=segment_params, 
    model_features=features, 
    target_col='CTR', 
    num_days=30
)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sim_df['days_since_launch'], sim_df['predicted_CTR'], marker='o', color='purple', linewidth=2)
plt.title("Simulated CTR Trajectory for Custom Creative")
plt.xlabel("Days Since Launch")
plt.ylabel("Predicted CTR")
plt.grid(True, alpha=0.3)
plt.show()