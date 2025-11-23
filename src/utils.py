
import csv, json, os

def load_pois(csv_path):
    try:
        with open(csv_path,'r') as f:
            return list(csv.DictReader(f))
    except:
        return []

def save_results(out_dir, trajectories, embeddings, labels):
    os.makedirs(out_dir, exist_ok=True)
    out = {
        'trajectories': trajectories,
        'embeddings': {str(k): v.tolist() for k,v in embeddings.items()},
        'labels': {str(i): int(l) for i,l in enumerate(labels)}
    }
    with open(os.path.join(out_dir, 'results.json'),'w') as f:
        json.dump(out, f)

