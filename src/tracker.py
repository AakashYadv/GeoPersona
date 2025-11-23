from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7
        )

    def update_tracks(self, bboxes, classes, scores, frame):

        if len(bboxes) == 0:
            return []

        # DeepSORT expects:
        #   [x1, y1, x2, y2, score]
        detections = []
        for box, score in zip(bboxes, scores):
            detections.append([
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(score)
            ])

        tracks = self.tracker.update_tracks(detections, frame=frame)

        return [t for t in tracks if t.is_confirmed()]
