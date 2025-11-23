import argparse
import os
import cv2

from src.detector import Detector
from src.tracker import Tracker
from src.embedding import MovementEncoder
from src.clustering import PersonaClustering
from src.utils import load_pois, save_results


# ---------------------------------------------------------
# MERGE DUPLICATE TRACK IDs (DeepSORT reassign issues)
# ---------------------------------------------------------
def merge_tracks_by_overlap(trajectories):
    merged = {}
    used = set()

    ids = list(trajectories.keys())

    for i in range(len(ids)):
        if ids[i] in used:
            continue

        base = ids[i]
        merged[base] = trajectories[base]
        used.add(base)

        for j in range(i + 1, len(ids)):
            if ids[j] in used:
                continue

            # Compare last point of base with first point of next
            if trajectories[base][-1][:2] == trajectories[ids[j]][0][:2]:
                merged[base] += trajectories[ids[j]]
                used.add(ids[j])

    return merged


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def run_pipeline(video_path, pois_csv, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    detector = Detector()
    tracker = Tracker()
    encoder = MovementEncoder()
    clusterer = PersonaClustering()

    pois = load_pois(pois_csv)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    trajectories = {}

    # ---------------------------------------------------------
    # FRAME-BY-FRAME TRACKING
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, classes, scores = detector.detect(frame)
        tracks = tracker.update_tracks(bboxes, classes, scores, frame)

        for t in tracks:
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_tlbr())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            trajectories.setdefault(tid, []).append((cx, cy, frame_idx))

        frame_idx += 1

    cap.release()

    # ---------------------------------------------------------
    # STEP 1: MERGE DUPLICATE IDs
    # ---------------------------------------------------------
    trajectories = merge_tracks_by_overlap(trajectories)

    # ---------------------------------------------------------
    # STEP 2: REMOVE SHORT TRACKLETS (< 10 FRAMES)
    # ---------------------------------------------------------
    trajectories = {
        tid: traj for tid, traj in trajectories.items()
        if len(traj) > 10
    }

    print(f"[INFO] Valid trajectories after cleanup: {len(trajectories)}")

    # ---------------------------------------------------------
    # STEP 3: GENERATE EMBEDDINGS
    # ---------------------------------------------------------
    embeddings = {
        tid: encoder.encode(traj)
        for tid, traj in trajectories.items()
    }

    # ---------------------------------------------------------
    # STEP 4: CLUSTER PERSONAS
    # ---------------------------------------------------------
    if len(embeddings) > 0:
        labels = clusterer.cluster(list(embeddings.values()))
    else:
        labels = []
        print("[WARNING] No embeddings to cluster.")

    # ---------------------------------------------------------
    # STEP 5: SAVE RESULTS
    # ---------------------------------------------------------
    save_results(output_dir, trajectories, embeddings, labels)

    print("[DONE] Pipeline completed successfully!")
    print(f"[RESULT] Saved at: {output_dir}")


# ---------------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to video")
    ap.add_argument("--pois", default="data/pois.csv")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    run_pipeline(args.input, args.pois, args.output)
