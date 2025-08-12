import json
import os

with open('annotation/blood_neutrophil.json', 'r') as f:
    annotations = json.load(f)

new_annotations = {}

for img_key, data in annotations.items():
    img_filename = f"{img_key}.png"
    new_annotations[img_filename] = []
    
    for pred in data.get('predictions', []):
        cx, cy = pred['x'], pred['y']
        w, h = pred['width'], pred['height']
        
        # center → (x_min, y_min, x_max, y_max)
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        
        new_annotations[img_filename].append({
            'bbox': [x_min, y_min, x_max, y_max],
            'score': pred.get('score'),
            'label': pred.get('label')
        })

output_path = 'annotation/blood_neutrophil_converted.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(new_annotations, f, indent=2)

print(f"변환 완료: {output_path}")
