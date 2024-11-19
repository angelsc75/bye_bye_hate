import os
import sys

def check_project_structure():
    required_files = [
        'src/app.py',
        'models/final_model.h5',
        'models/tokenizer.pickle',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print("✅ All required files are present")
    return True

if __name__ == "__main__":
    if not check_project_structure():
        sys.exit(1)