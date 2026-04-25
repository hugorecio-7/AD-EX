import subprocess
import os
import threading
import time
import sys

# Detect the project's virtual environment python
root_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt': # Windows
    venv_python = os.path.join(root_dir, '.venv', 'Scripts', 'python.exe')
else:
    venv_python = os.path.join(root_dir, '.venv', 'bin', 'python')

# Fallback if venv is not found
if not os.path.exists(venv_python):
    print(f"[SYSTEM] Warning: .venv not found at {venv_python}. Using default python.")
    venv_python = sys.executable

def run_backend():
    print("[SYSTEM] Starting Backend API...")
    # Use the explicit venv python to ensure all dependencies are found
    backend_dir = os.path.join(root_dir, 'backend')
    backend_script = 'main.py'
    subprocess.run([venv_python, backend_script], cwd=backend_dir)

def run_frontend():
    print("[SYSTEM] Starting Frontend Dev Server...")
    frontend_dir = os.path.join(root_dir, 'frontend')
    if os.path.exists(frontend_dir):
        os.chdir(frontend_dir)
        subprocess.run('npm run dev', shell=True)
    else:
        print(f"[SYSTEM] Error: Frontend directory not found at {frontend_dir}")

if __name__ == "__main__":
    print("="*50)
    print("      SMADEX INTELLIGENCE SYSTEM BOOT")
    print("="*50)
    
    # 1. Start Backend in a background thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()

    # Give backend a second to print its start message
    time.sleep(2)
    
    # 2. Start Frontend
    run_frontend()
