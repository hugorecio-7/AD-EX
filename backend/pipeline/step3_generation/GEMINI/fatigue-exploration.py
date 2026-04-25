
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_theme(style="whitegrid")

def plot_status_comparison(stats_df, summary_df, samples_per_status=3):
    """
    Samples an equal number of creatives from each status ('fatigued', 'stable', null)
    and plots their ground-truth CTR with fatigue lines to compare tendencies.
    """
    print(f"--- Sampling {samples_per_status} Creatives per Status Category ---")
    
    # 1. Standardize the status column to make it easy to filter
    # Pandas treats null/empty strings as NaNs, so we fill them with a string label
    summary_df = summary_df.copy()
    summary_df['creative_status'] = summary_df['creative_status'].fillna('no_status')
    
    # Define our target groups (handling varying casing just in case)
    summary_df['creative_status'] = summary_df['creative_status'].str.lower()
    statuses = ['fatigued', 'stable', 'no_status'] 
    
    # 2. Sample the creatives evenly across statuses
    sampled_creatives = []
    
    for status in statuses:
        # Isolate creatives that belong to this specific status
        pool = summary_df[summary_df['creative_status'] == status]['creative_id'].unique()
        
        # Sample up to the requested amount (fallback if there are fewer than 5 in a category)
        n_samples = min(samples_per_status, len(pool))
        
        if n_samples > 0:
            samples = np.random.choice(pool, size=n_samples, replace=False)
            for cid in samples:
                sampled_creatives.append({'creative_id': cid, 'display_status': status.capitalize()})
    
    if not sampled_creatives:
        print("No creatives found matching the statuses. Check your column names and values!")
        return
        
    sampled_df_refs = pd.DataFrame(sampled_creatives)
    
    # 3. Merge the selected samples with our daily stats and summary info
    # We bring in both fatigue_day and creative_status from the summary
    plot_base_df = pd.merge(
        stats_df[stats_df['creative_id'].isin(sampled_df_refs['creative_id'])], 
        summary_df[['creative_id', 'fatigue_day', 'creative_status']], 
        on='creative_id', 
        how='left'
    )
    
    # Calculate CTR
    eps = 1e-5
    plot_base_df['CTR'] = plot_base_df['clicks'] / (plot_base_df['impressions'] + eps)
    
    # 4. Setup the Plot Grid
    num_plots = len(sampled_creatives)
    num_cols = 3  # 3 columns creates a nice wide layout for time series
    num_rows = math.ceil(num_plots / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, row in sampled_df_refs.iterrows():
        ax = axes[idx]
        c_id = row['creative_id']
        c_status = row['display_status']
        
        # Isolate the data for this specific creative
        creative_data = plot_base_df[plot_base_df['creative_id'] == c_id].copy()
        
        # Pick the most prominent segment (Country/OS) to keep the time-series line smooth
        top_country = creative_data['country'].mode()[0]
        top_os = creative_data['os'].mode()[0]
        plot_data = creative_data[(creative_data['country'] == top_country) & (creative_data['os'] == top_os)].copy()
        
        # Sort strictly by chronological order
        plot_data = plot_data.sort_values('days_since_launch').reset_index(drop=True)
        
        # Extract the true fatigue day (if it exists)
        fatigue_day = plot_data['fatigue_day'].iloc[0] if not plot_data.empty else np.nan
        
        # Plot the Actual CTR
        if not plot_data.empty:
            # Color code based on status for even faster visual scanning
            line_color = 'darkorange' if c_status == 'Fatigued' else ('seagreen' if c_status == 'Stable' else 'gray')
            
            ax.plot(plot_data['days_since_launch'], plot_data['CTR'], marker='o', 
                    label='Actual Daily CTR', color=line_color, linewidth=2)
            
            # Add the Red Fatigue Line if it actually fatigued
            if pd.notna(fatigue_day):
                ax.axvline(x=fatigue_day, color='red', linestyle='--', linewidth=2.5, 
                           label=f'Fatigue (Day {int(fatigue_day)})')
            else:
                ax.plot([], [], ' ', label='No Fatigue Detected')
        
        # Formatting the subplot
        ax.set_title(f'[{c_status}] ID: {c_id}\n{top_country} - {top_os}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Days Since Launch')
        ax.set_ylabel('CTR')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Clean up any unused subplots (e.g., if you asked for 5 per status but only had 4 stables)
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

# --- Execution ---

DATA = "../../frontend/public/data/"
# Load your files
stats_df = pd.read_csv(f'{DATA}creative_daily_country_os_stats.csv')
summary_df = pd.read_csv(f'{DATA}creative_summary.csv')

# Run the visualization
plot_status_comparison(stats_df, summary_df)