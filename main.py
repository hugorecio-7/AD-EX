import subprocess
import os
import threading
import time

def run_backend():
    print("[SYSTEM] Starting Backend API...")
    # Change dir to backend so relative imports work
    os.chdir('backend')
    subprocess.run(['python', 'main.py'])

def run_frontend():
    print("[SYSTEM] Starting Frontend Dev Server...")
    os.chdir('frontend')
    # On Windows we use shell=True for npm
    subprocess.run('npm run dev', shell=True)

if __name__ == "__main__":
    print("="*50)
    print("      SMADEX INTELLIGENCE SYSTEM BOOT")
    print("="*50)
    
    # Store the original root path
    root_path = os.getcwd()

    # 1. Start Backend in a background thread
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()

    # Give backend a second to print its start message
    time.sleep(1)
    
    # Reset dir before starting next
    os.chdir(root_path)

    # 2. Start Frontend (Blocks main thread so we can keep the console open)
    run_frontend()
