import subprocess
import sys
import os

def run_script(path):
    print(f"\n>>> Running {path}...")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"!!! Error running {path}")
    else:
        print(f">>> {path} completed successfully.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 0. Build the retrieval scoring index (Performance, Health, Confidence scores)
    run_script(os.path.join(script_dir, "preprocess_retrieval_index.py"))
    
    # 1. Preprocess Scores (Initializes data.json and performance_scores.csv)
    run_script(os.path.join(script_dir, "preprocess_scores.py"))
    
    # 2. Preprocess Masks (SAM + OCR for all 1080 images)
    run_script(os.path.join(script_dir, "preprocess_masks.py"))
    
    print("\n[FINISH] All preprocessing steps completed!")
