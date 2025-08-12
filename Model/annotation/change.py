import json
from datetime import datetime

INPUT_JSON = 'blood_neutrophil_converted.json'  

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    raw = json.load(f)

images = []
annotations = []
ann_id = 1

for img_idx, (fname, items) in enumerate(raw.items(), start=1):
    images.append({
        "id": img_idx,
        "file_name": fname,
    })
    for obj in items:
        x1, y1, x2, y2 = obj["bbox"]
        w = x2 - x1
        h = y2 - y1
        annotations.append({
            "id": ann_id,
            "image_id": img_idx,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
            "category_id": 0,  
        })
        ann_id += 1

coco = {
    "info": {
        "description": "My COCO Dataset",
        "url": "",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": [
        {
            "id": 0,
            "name": "neutrophil",       
            "supercategory": "cell"
        }
    ]
}

with open('coco_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print("coco_dataset.json 생성 완료")
