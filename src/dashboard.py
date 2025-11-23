import streamlit as st
import json
import cv2
import numpy as np
from collections import defaultdict
import math

st.set_page_config(layout="wide")
st.title("GeoPersona — Minimalist Dashboard with Alerts")

# ---------- CONFIG ----------
VIDEO_PATH = "data/videos/mall.mp4"   # change to your video file
RESULTS_PATH = "results/results.json"

# behavior & dwell thresholds
MIN_MOVEMENT_FOR_DWELL = 2.0        # px threshold to consider "moving"
LOITERING_SECONDS = 5.0             # loitering alert threshold
FAST_MOVER_SPEED_PX_S = 300.0       # fast-mover alert threshold

# heatmap config
GAUSSIAN_BLUR_KSIZE = (51, 51)
HEATMAP_ALPHA = 0.6

# crowding config
CROWDING_THRESHOLD_RATIO = 0.02   # fraction of pixels considered "crowded" to raise alert
CROWDING_PIXEL_THRESHOLD = 0.65   # relative threshold per-pixel against max heat

# Restricted zones (x1, y1, x2, y2) in pixel coordinates — edit for your video
RESTRICTED_ZONES = [
    {"name": "Staff Only", "coords": (100, 100, 400, 400)},
    {"name": "Exit Area", "coords": (1500, 100, 1900, 400)}
]

# expected main flow direction (for reverse flow detection): "Right" or "Left"
EXPECTED_FLOW_DIRECTION = "Right"
REVERSE_FLOW_MIN_DELTA_X = 50  # px threshold to consider significant left/right movement

# ---------- LOAD RESULTS ----------
try:
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    trajectories = results.get("trajectories", {})
    raw_labels = results.get("labels", {})
except Exception:
    st.error("Could not load results.json. Run pipeline first.")
    st.stop()

labels = {str(k): int(v) for k, v in raw_labels.items()} if isinstance(raw_labels, dict) else {}

# ---------- UTILITIES ----------
def deterministic_color(i):
    rng = np.random.RandomState(int(i) + 12345)
    rgb = rng.randint(50, 230, size=3).tolist()
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_cluster_color_map(labels_dict):
    unique = sorted({v for v in labels_dict.values()})
    color_map = {cid: deterministic_color(cid) for cid in unique}
    color_map[-1] = (200, 200, 200)  # fallback grey for unknown cluster
    return color_map

def compute_total_distance(traj):
    if len(traj) < 2:
        return 0.0
    pts = np.array([[p[0], p[1]] for p in traj], dtype=float)
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

def compute_avg_speed_px_per_sec(traj, fps):
    if len(traj) < 2 or fps <= 0:
        return 0.0
    total_dist = compute_total_distance(traj)
    time_seconds = (traj[-1][2] - traj[0][2]) / fps
    return total_dist / time_seconds if time_seconds > 0 else 0.0

def estimate_dwell_time(traj, fps, move_thresh=MIN_MOVEMENT_FOR_DWELL):
    if len(traj) < 2 or fps <= 0:
        return 0.0
    dwell_frames = 0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i-1][0]
        dy = traj[i][1] - traj[i-1][1]
        if math.hypot(dx, dy) < move_thresh:
            dwell_frames += 1
    return dwell_frames / fps

# ---------- BEHAVIOR CLASSIFICATION ----------
def classify_behavior(traj, fps):
    total_dist = compute_total_distance(traj)
    avg_speed = compute_avg_speed_px_per_sec(traj, fps)
    dwell = estimate_dwell_time(traj, fps)

    if avg_speed > FAST_MOVER_SPEED_PX_S and dwell < 1:
        return "Fast-Mover"
    if avg_speed < 50 and dwell > 3 and total_dist < 200:
        return "Loiterer"
    if 50 <= avg_speed <= FAST_MOVER_SPEED_PX_S:
        return "Wanderer"
    return "Unknown"

# ---------- ALERT HELPERS ----------
def check_loitering(traj, fps, threshold_sec=LOITERING_SECONDS):
    dwell = estimate_dwell_time(traj, fps)
    return dwell > threshold_sec, dwell

def check_fast_mover(traj, fps, speed_threshold=FAST_MOVER_SPEED_PX_S):
    speed = compute_avg_speed_px_per_sec(traj, fps)
    return speed > speed_threshold, speed

def point_in_zone(x, y, zone):
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def check_restricted_entry(traj, restricted_zones):
    if len(traj) == 0:
        return False, None
    x, y, _ = traj[-1]  # last known point
    for z in restricted_zones:
        if point_in_zone(x, y, z["coords"]):
            return True, z["name"]
    return False, None

def compute_flow_direction(traj):
    if len(traj) < 5:
        return None
    start_x = traj[0][0]
    end_x = traj[-1][0]
    delta = end_x - start_x
    if delta > REVERSE_FLOW_MIN_DELTA_X:
        return "Right"
    if delta < -REVERSE_FLOW_MIN_DELTA_X:
        return "Left"
    return "Neutral"

def check_reverse_flow(traj, expected=EXPECTED_FLOW_DIRECTION):
    flow = compute_flow_direction(traj)
    if flow is None or flow == "Neutral":
        return False, flow
    return (flow != expected), flow

def compute_crowding_level(heatmap, pixel_threshold_ratio=CROWDING_PIXEL_THRESHOLD):
    if heatmap.max() == 0:
        return 0.0
    total_pixels = heatmap.size
    threshold_value = heatmap.max() * pixel_threshold_ratio
    crowded_pixels = np.sum(heatmap > threshold_value)
    return float(crowded_pixels) / float(total_pixels)

# ---------- LOAD VIDEO FIRST FRAME ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    st.error(f"Cannot open video: {VIDEO_PATH}")
    st.stop()

ret, first_frame = cap.read()
if not ret:
    st.error("Cannot read first frame of the video.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_h, frame_w = first_frame.shape[:2]
cap.release()

first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
cluster_color_map = get_cluster_color_map(labels)

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["Trajectories", "Heatmap", "Insights", "Alerts"])

# -------------------- TAB 1: Trajectories --------------------
with tab1:
    st.header("Trajectories (cluster-colored)")
    canvas = first_frame_rgb.copy()

    for tid, pts in trajectories.items():
        cid = labels.get(str(tid), -1)
        color = cluster_color_map.get(cid, (200, 200, 200))

        # draw path lines
        for i in range(1, len(pts)):
            x1, y1, _ = pts[i-1]
            x2, y2, _ = pts[i]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # draw last point and label
        if len(pts) >= 1:
            lx, ly, _ = pts[-1]
            cv2.circle(canvas, (int(lx), int(ly)), 5, color, -1)
            cv2.putText(canvas, f"ID{tid} C{cid}", (int(lx)+6, int(ly)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # draw restricted zones overlay
    for z in RESTRICTED_ZONES:
        x1, y1, x2, y2 = z["coords"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, z["name"], (x1+6, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    st.image(canvas, use_column_width=True)

# -------------------- TAB 2: Heatmap --------------------
with tab2:
    st.header("Movement Heatmap (Gaussian-blurred)")
    heat = np.zeros((frame_h, frame_w), dtype=np.float32)

    for _, pts in trajectories.items():
        for x, y, _ in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < frame_w and 0 <= yi < frame_h:
                heat[yi, xi] += 1.0

    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
        heat_blurred = cv2.GaussianBlur(heat_norm, GAUSSIAN_BLUR_KSIZE, 0)
        heat_colored = cv2.applyColorMap(heat_blurred, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(first_frame_rgb, 1.0 - HEATMAP_ALPHA, heat_colored, HEATMAP_ALPHA, 0)
        st.image(overlay, use_column_width=True)
    else:
        st.info("No trajectories to build heatmap.")

# -------------------- TAB 3: Insights --------------------
with tab3:
    st.header("Insights")
    total_tracks = len(trajectories)
    st.write(f"Total Valid Tracklets: **{total_tracks}**")

    # cluster summary
    cluster_summary = defaultdict(lambda: {"count": 0, "total_distance": 0.0, "total_speed": 0.0, "total_dwell": 0.0})
    for tid, pts in trajectories.items():
        cid = labels.get(str(tid), -1)
        dist = compute_total_distance(pts)
        speed = compute_avg_speed_px_per_sec(pts, fps)
        dwell = estimate_dwell_time(pts, fps)

        cluster_summary[cid]["count"] += 1
        cluster_summary[cid]["total_distance"] += dist
        cluster_summary[cid]["total_speed"] += speed
        cluster_summary[cid]["total_dwell"] += dwell

    st.subheader("Cluster Summary")
    for cid, stats in sorted(cluster_summary.items()):
        cnt = stats["count"]
        avg_dist = stats["total_distance"] / cnt if cnt else 0.0
        avg_speed = stats["total_speed"] / cnt if cnt else 0.0
        avg_dwell = stats["total_dwell"] / cnt if cnt else 0.0
        st.write(f"Cluster C{cid}: Count={cnt}, AvgDist={avg_dist:.1f}px, AvgSpeed={avg_speed:.1f}px/s, AvgDwell={avg_dwell:.1f}s")

    st.markdown("---")

    st.subheader("Behavior Classification (per track)")
    for tid, pts in trajectories.items():
        behavior = classify_behavior(pts, fps)
        dist = compute_total_distance(pts)
        speed = compute_avg_speed_px_per_sec(pts, fps)
        dwell = estimate_dwell_time(pts, fps)
        st.write(f"ID {tid}: **{behavior}** — Dist {dist:.1f}px | Speed {speed:.1f}px/s | Dwell {dwell:.1f}s")

# -------------------- TAB 4: Alerts --------------------
with tab4:
    st.header("Real-Time Alerts")

    alerts = []

    # Per-person alerts
    for tid, pts in trajectories.items():
        # Loitering detection
        loitering, dwell = check_loitering(pts, fps)
        if loitering:
            alerts.append({"entity": f"Person {tid}", "type": "Loitering", "info": f"dwell={dwell:.1f}s", "severity": 2})

        # Fast mover detection
        fast, speed = check_fast_mover(pts, fps)
        if fast:
            alerts.append({"entity": f"Person {tid}", "type": "Fast-Mover", "info": f"speed={speed:.1f}px/s", "severity": 2})

        # Restricted zone entry
        in_zone, zone_name = check_restricted_entry(pts, RESTRICTED_ZONES)
        if in_zone:
            alerts.append({"entity": f"Person {tid}", "type": "Restricted Area Entry", "info": f"zone={zone_name}", "severity": 3})

        # Reverse flow detection
        reversed_flow, flow_dir = check_reverse_flow(pts, expected=EXPECTED_FLOW_DIRECTION)
        if reversed_flow:
            info = f"flow={flow_dir} != expected {EXPECTED_FLOW_DIRECTION}"
            alerts.append({"entity": f"Person {tid}", "type": "Reverse Flow", "info": info, "severity": 1})

    # Crowding alert (scene-level)
    heat = np.zeros((frame_h, frame_w), dtype=np.float32)
    for _, pts in trajectories.items():
        for x, y, _ in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < frame_w and 0 <= yi < frame_h:
                heat[yi, xi] += 1.0

    crowd_ratio = compute_crowding_level(heat, pixel_threshold_ratio=CROWDING_PIXEL_THRESHOLD)
    if crowd_ratio > CROWDING_THRESHOLD_RATIO:
        alerts.append({"entity": "Scene", "type": "Crowding", "info": f"density_ratio={crowd_ratio:.3f}", "severity": 3})

    # Sort alerts by severity descending
    alerts = sorted(alerts, key=lambda x: x["severity"], reverse=True)

    if len(alerts) == 0:
        st.success("No alerts detected.")
    else:
        st.warning(f"{len(alerts)} alerts detected")
        for a in alerts:
            st.markdown(f"**{a['type']}** — {a['entity']} — {a['info']}")
            # show colored box for severity
            if a["severity"] >= 3:
                st.markdown(f"<div style='padding:8px;background:#ffcccc;border-left:4px solid #cc0000'>High severity</div>", unsafe_allow_html=True)
            elif a["severity"] == 2:
                st.markdown(f"<div style='padding:6px;background:#fff4cc;border-left:4px solid #ffcc00'>Medium severity</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding:4px;background:#e8f7ff;border-left:4px solid #0099ff'>Low severity</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Alerts are generated from simple rule-based heuristics. Tune thresholds in the CONFIG section for your scene/video.")
