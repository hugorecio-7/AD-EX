import os
import sys
import json
from pathlib import Path

# Afegir el directori arrel al PATH de Python perquè pugui importar 'pipeline'
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# Importem directament del teu pipeline
from pipeline.step2_feature_analysis.llm_feature_gap import analyze_feature_gap_with_llm

def main():
    print("🚀 Iniciant Test del Pipeline: Vision Feature Gap Analysis")
    
    # Pots canviar aquests IDs pels que tinguis a 'output/features/creative_XXX/'
    original_ad_id = "500000"
    top_ads_ids = ["500001", "500002"] # Podries posar-ne varis: ["500000", "500001"]

    print(f"\n🔍 Comparant Anunci Original ({original_ad_id}) amb Top Performers ({top_ads_ids})...")
    print("🤖 Això utilitzarà GPT-4o llegint les imatges i els JSONs estructurats...\n")

    # Truquem a la funció del TEU pipeline directament
    result = analyze_feature_gap_with_llm(
        query_creative_id=original_ad_id,
        top_ids=top_ads_ids,
        max_features=5
    )

    print("\n" + "="*50)
    print("✨ RESULTATS DEL FEATURE GAP:")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()  