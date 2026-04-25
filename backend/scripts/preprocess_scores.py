import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir)

import csv
import json
from pipeline.step3_generation.core import evaluate_dynamic_creative

def clean_str(val):
    if val is None: return ""
    return str(val).strip()

def csv_to_json():
    # Detect current directory and adjust paths
    base_dir = os.getcwd()
    if 'backend' in base_dir:
        root_dir = base_dir.split('backend')[0]
    else:
        root_dir = base_dir

    creatives_path = os.path.join(root_dir, 'frontend/public/data/creatives.csv')
    summary_path = os.path.join(root_dir, 'frontend/public/data/creative_summary.csv')
    output_path = os.path.join(root_dir, 'frontend/src/mocks/data.json')
    scores_csv_path = os.path.join(root_dir, 'frontend/public/data/performance_scores.csv')
    assets_dir = os.path.join(root_dir, 'frontend/public/data/assets')

    # Map summary data by creative_id
    summary_map = {}
    with open(summary_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = clean_str(row.get('creative_id'))
            summary_map[cid] = row

    data = []
    with open(creatives_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = clean_str(row.get('creative_id'))
            
            # Map columns (Cleanly named CSV)
            adv_name = clean_str(row.get('advertiser_name'))
            format_val = clean_str(row.get('format'))
            theme_val = clean_str(row.get('theme'))
            hook_val = clean_str(row.get('hook_type'))
            headline = clean_str(row.get('headline'))
            
            summary = summary_map.get(cid, {})
            raw_ctr = summary.get('overall_ctr', '0')
            try:
                ctr = round(float(raw_ctr) * 100, 2)
            except:
                ctr = 0.0

            # Run the dynamic score regressor
            features = [f for f in [format_val, theme_val, hook_val] if f]
            analysis = evaluate_dynamic_creative(None, features)
            
            # Base it a bit on the actual CTR to make mock data look realistic
            base_perf = min(0.99, max(0.01, (ctr / 6.0)))
            perf_score = round((analysis["performance_score"] + base_perf) / 2, 3)
            is_fatigued = perf_score < 0.3

            # Dynamically check for high-fidelity assets in data/assets/
            asset_filename = f"creative_{cid}.png"
            asset_path = os.path.join(assets_dir, asset_filename)
            
            if os.path.exists(asset_path):
                image_url = f"/data/assets/{asset_filename}"
            else:
                image_url = 'https://images.unsplash.com/photo-1518770660439-4636190af475'

            entry = {
                "id": cid,
                "campaign": clean_str(row.get('campaign_id')),
                "advertiser": adv_name,
                "format": format_val,
                "theme": theme_val,
                "hook_type": hook_val,
                "performance_score": perf_score,
                "image_url": image_url,
                "ctr": ctr,
                "cpi": round(1.0 + (float(cid) % 100) / 100, 2),
                "fatigued": is_fatigued,
                "insights": headline or f"AI Performance analysis: {perf_score}.",
                "cluster_id": f"{format_val}-{theme_val}-{hook_val}"
            }
            data.append(entry)

    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # Save a separate backend score file (CSV) as requested
    with open(scores_csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['creative_id', 'performance_score', 'is_fatigued'])
        for entry in data:
            writer.writerow([entry['id'], entry['performance_score'], entry['fatigued']])
    
    print(f"[SUCCESS] Generated mock data to {output_path}")
    print(f"[SUCCESS] Generated backend scores to {scores_csv_path}")

if __name__ == "__main__":
    csv_to_json()
