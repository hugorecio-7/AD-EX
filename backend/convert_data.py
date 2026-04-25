import csv
import json
import os

def clean_str(val):
    if val is None: return ""
    return str(val).strip()

def csv_to_json():
    # Paths assuming script runs from root or relevant dir
    creatives_path = 'data/creatives.csv'
    summary_path = 'data/creative_summary.csv'
    output_path = 'frontend/src/mocks/data.json'

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
            
            # Map columns (handling the space-padded names from the CSV)
            adv_name = clean_str(row.get(' advertiser_name'))
            format_val = clean_str(row.get(' format        '))
            theme_val = clean_str(row.get(' theme        '))
            hook_val = clean_str(row.get(' hook_type       '))
            headline = clean_str(row.get(' headline                  '))
            
            summary = summary_map.get(cid, {})
            # Normalized overall_ctr (e.g. 0.0038 -> 0.38 for display)
            raw_ctr = summary.get(' overall_ctr          ', '0')
            try:
                ctr = round(float(raw_ctr) * 100, 2)
            except:
                ctr = 0.0

            fatigue = clean_str(summary.get(' fatigue_day', '')).strip()
            is_fatigued = fatigue != ""

            # Dynamically check for high-fidelity assets in data/assets/
            # NOTE: Filenames in directory are creative_500000.png, but creative_id might be 500000
            asset_filename = f"creative_{cid}.png"
            asset_path = os.path.join('data', 'assets', asset_filename)
            
            # If the specific asset exists, use it. Otherwise, use a generic vertical placeholder.
            if os.path.exists(asset_path):
                # We point to /data/assets/... which we will serve via Vite static config or similar
                # For now, let's assume Vite can serve the 'data' folder
                image_url = f"/data/assets/{asset_filename}"
            else:
                image_url = 'https://images.unsplash.com/photo-1518770660439-4636190af475'

            entry = {
                "id": cid,
                "campaign": clean_str(row.get(' campaign_id')),
                "advertiser": adv_name,
                "format": format_val,
                "theme": theme_val,
                "hook_type": hook_val,
                "image_url": image_url,
                "ctr": ctr,
                "cpi": round(1.0 + (float(cid) % 100) / 100, 2), # Mocking CPI
                "fatigued": is_fatigued,
                "insights": headline or f"Performance score: {summary.get(' perf_score', 'N/A')}.",
                "cluster_id": f"{format_val}-{theme_val}-{hook_val}"
            }
            data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    csv_to_json()
