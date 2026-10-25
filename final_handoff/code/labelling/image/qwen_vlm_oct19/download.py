"""Download one HF model snapshot into the default cache. Usage: python download.py REPO_ID"""
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.py"]), flush=True)
