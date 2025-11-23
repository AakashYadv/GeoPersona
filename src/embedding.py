
import numpy as np
from sklearn.preprocessing import StandardScaler

class MovementEncoder:
    def __init__(self):
        self.scaler = StandardScaler()

    def trajectory_to_features(self, traj):
        xs = np.array([p[0] for p in traj])
        ys = np.array([p[1] for p in traj])
        ts = np.array([p[2] for p in traj])
        deltas = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        speeds = deltas / (np.diff(ts) + 1e-6)

        features = [
            np.mean(speeds) if len(speeds) else 0,
            np.std(speeds) if len(speeds) else 0,
            np.sum(deltas) if len(deltas) else 0,
            np.sum(speeds < 1.0) if len(speeds) else 0
        ]
        return np.array(features)

    def encode(self, traj):
        return self.trajectory_to_features(traj)

