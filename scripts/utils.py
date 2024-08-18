import os

def create_directories():
    os.makedirs("data/videos", exist_ok=True)
    os.makedirs("frames", exist_ok=True)
    os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    create_directories()
