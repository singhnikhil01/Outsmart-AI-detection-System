import os
import subprocess

def install(package):
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])

def main():
    required_packages = [
        "torch",
        "datasets",
        "transformers",
        "trl",
        "tqdm",
        "nltk",
        "language_tool_python"
    ]
    
    for package in required_packages:
        install(package)

def run_script(script_name):
    subprocess.check_call([os.sys.executable, script_name])

if __name__ == "__main__":
    main()
    run_script('train.py')
