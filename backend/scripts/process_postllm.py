import os
import sys
import subprocess

# ==========================================
# 1. CONFIGURACIÓ DE RUTES
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(backend_dir)

# Ruta on buscarem les carpetes ja pre-processades
features_dir = os.path.join(project_root, "output", "features")

# Ruta del teu script LLM
postllm_script = os.path.join(project_root, "backend", "generate", "postllm.py")

def main():
    print("🚀 Iniciant processament per lots (Batch) de Creatives...")
    
    if not os.path.exists(features_dir):
        print(f"❌ Error: No s'ha trobat el directori {features_dir}")
        return

    if not os.path.exists(postllm_script):
        print(f"❌ Error: No s'ha trobat el script {postllm_script}")
        return

    # Llistar totes les carpetes que comencin per "creative_"
    carpetes = [d for d in os.listdir(features_dir) 
                if os.path.isdir(os.path.join(features_dir, d)) and d.startswith("creative_")]
    
    if not carpetes:
        print("⚠️ No s'han trobat carpetes de creatives per processar a output/features.")
        return

    print(f"📂 S'han detectat {len(carpetes)} creatives. Començant iteració...\n")

    # Agafem el python de l'entorn virtual actual (.venv)
    python_exe = sys.executable

    # ==========================================
    # 2. BUCLE PER CADA CREATIVE
    # ==========================================
    for carpeta in carpetes:
        creative_id = carpeta.replace("creative_", "")
        
        # Comprovar si ja s'ha generat l'estructura prèviament per saltar-ho i estalviar tokens
        output_file = os.path.join(features_dir, carpeta, f"creative_{creative_id}_structured.json")
        if os.path.exists(output_file):
            # Print explícit dient que ja està fet i se'l salta
            print(f"⏭️  Saltant ID: {creative_id} -> Ja està fet.")
            continue

        # Print explícit abans d'enviar a l'LLM
        print("\n" + "="*50)
        print(f"🧠 Analitzant ID: {creative_id} amb LLMs...")
        print("="*50)
        
        try:
            # Executem el script postllm.py passant-li l'ID al final
            subprocess.run([python_exe, postllm_script, creative_id], check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executant postllm.py per al creative {creative_id}: {e}")
            print("Continuant amb el següent...")
            continue

    print("\n✅ Processament per lots finalitzat amb èxit!")

if __name__ == "__main__":
    main()