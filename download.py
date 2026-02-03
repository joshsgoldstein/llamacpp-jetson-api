import os
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()
model_id = os.getenv("MODEL_ID")
local_model_path = f"./models/{model_id}"

# 🔹 Ensure directory exists
os.makedirs(local_model_path, exist_ok=True)

# 1. Download model repo locally if it doesn't exist yet
if not os.path.isdir(local_model_path) or not os.listdir(local_model_path):
    print(f"📦 Downloading model to {local_model_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,  # avoids dangling symlinks on some FS
    )
    print("✅ Download complete.")
else:
    print(f"✅ Model already exists at {local_model_path}")