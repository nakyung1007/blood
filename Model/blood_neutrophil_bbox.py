import os
import json
from inference_sdk import InferenceHTTPClient

from config.paths import API_URL, API_KEY, WORKSPACE, WORKFLOW

api_url=API_URL
api_key = API_KEY
workspace= WORKSPACE
workflow = WORKFLOW

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

img_path = "IMG_PATH"

result = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW,
    images={"image": img_path},
    use_cache=True
)

try:
    payload = result.json()
except AttributeError:
    payload = result

if isinstance(payload, list) and len(payload) > 0:
    entry = payload[0]
else:
    entry = payload

preds = entry.get("predictions", entry)

out_dir  = os.path.join("annotation")
out_file = os.path.join(out_dir, "blood_neutrophil.json")
os.makedirs(out_dir, exist_ok=True)

if os.path.exists(out_file):
    with open(out_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
else:
    all_data = {}

base_name = os.path.splitext(os.path.basename(img_path))[0]

all_data[base_name] = preds

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
