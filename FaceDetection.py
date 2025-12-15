import argparse
import os
import time
import math
import collections
from typing import List, Tuple, Dict, Optional

import cv2  # import openCV

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected to be available in runtime env
    np = None  # type: ignore


# ------------------------------
# Argument parsing
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Face detection using OpenCV Haar cascades with optional recognition, tracking, and landmarks")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: camera index (e.g., 0), path to video file, or path to image")
    parser.add_argument("--cascade", type=str, default="haarcascade_frontalface_default.xml",
                        help="Path to Haar cascade xml file")
    parser.add_argument("--scale-factor", type=float, default=1.1,
                        help="Parameter specifying how much the image size is reduced at each image scale")
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="Parameter specifying how many neighbors each candidate rectangle should have to retain it")
    parser.add_argument("--min-size", type=int, nargs=2, metavar=("W", "H"), default=(30, 30),
                        help="Minimum possible object size. Objects smaller than that are ignored")
    parser.add_argument("--save-video", type=str, default="",
                        help="Optional path to save output video (MP4 or AVI based on extension)")
    parser.add_argument("--output-fps", type=float, default=30.0,
                        help="FPS for saved video if --save-video is set")
    parser.add_argument("--display", action="store_true",
                        help="Show window with detections")

    # Optional advanced features
    parser.add_argument("--track", action="store_true", help="Enable simple centroid-based multi-face tracking")

    # Recognition options (requires OpenCV contrib module cv2.face)
    parser.add_argument("--recognition-model", type=str, default="",
                        help="Path to LBPH recognition model (.yml). If provided, enables recognition")
    parser.add_argument("--train-recognition", type=str, default="",
                        help="Directory of training images organized as person_name/imagename.jpg to train LBPH recognizer")
    parser.add_argument("--save-recognition-model", type=str, default="",
                        help="Optional path to save the trained recognition model (.yml)")

    # Landmark options (requires OpenCV contrib Facemark LBF model)
    parser.add_argument("--landmarks-model", type=str, default="",
                        help="Path to LBF landmarks model (lbfmodel.yaml). If available, landmarks will be drawn")
    parser.add_argument("--align", action="store_true",
                        help="Align faces (requires landmarks)")

    return parser.parse_args()


# ------------------------------
# Utilities and helpers
# ------------------------------

def resolve_source(src_arg: str):
    # Try to interpret as int camera index first
    if src_arg.isdigit():
        return int(src_arg), True
    # Else assume it is a path
    return src_arg, False


def open_capture(source):
    cap = cv2.VideoCapture(source)
    return cap


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# ------------------------------
# Simple centroid tracker (dependency-free)
# ------------------------------

class CentroidTracker:
    def __init__(self, max_disappeared: int = 15, distance_thresh: float = 50.0):
        self.next_id = 1
        self.objects: Dict[int, Tuple[int, int, int, int]] = {}
        self.centroids: Dict[int, Tuple[float, float]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.distance_thresh = distance_thresh

    @staticmethod
    def _centroid(rect: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = rect
        return x + w / 2.0, y + h / 2.0

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        if len(rects) == 0:
            # mark disappeared
            to_delete = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    to_delete.append(oid)
            for oid in to_delete:
                self.objects.pop(oid, None)
                self.centroids.pop(oid, None)
                self.disappeared.pop(oid, None)
            return self.objects

        # if no tracked objects, register all
        if len(self.objects) == 0:
            for r in rects:
                cid = self.next_id
                self.next_id += 1
                self.objects[cid] = r
                self.centroids[cid] = self._centroid(r)
                self.disappeared[cid] = 0
            return self.objects

        # build cost matrix of distances between existing and new
        object_ids = list(self.objects.keys())
        object_centroids = [self.centroids[oid] for oid in object_ids]
        input_centroids = [self._centroid(r) for r in rects]

        # compute pairwise distances
        dists = []
        for oc in object_centroids:
            row = [math.hypot(oc[0] - ic[0], oc[1] - ic[1]) for ic in input_centroids]
            dists.append(row)

        # greedy matching by min distance
        used_rows = set()
        used_cols = set()
        matches = []
        while True:
            min_val = float('inf'); min_row = -1; min_col = -1
            for r_idx, row in enumerate(dists):
                if r_idx in used_rows:
                    continue
                for c_idx, val in enumerate(row):
                    if c_idx in used_cols:
                        continue
                    if val < min_val:
                        min_val = val; min_row = r_idx; min_col = c_idx
            if min_row == -1:
                break
            if min_val > self.distance_thresh:
                break
            used_rows.add(min_row)
            used_cols.add(min_col)
            matches.append((min_row, min_col))

        # update matched
        for r_idx, c_idx in matches:
            oid = object_ids[r_idx]
            rect = rects[c_idx]
            self.objects[oid] = rect
            self.centroids[oid] = input_centroids[c_idx]
            self.disappeared[oid] = 0

        # handle unmatched
        unmatched_rows = set(range(len(object_ids))) - used_rows
        for r_idx in unmatched_rows:
            oid = object_ids[r_idx]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.objects.pop(oid, None)
                self.centroids.pop(oid, None)
                self.disappeared.pop(oid, None)

        unmatched_cols = set(range(len(rects))) - used_cols
        for c_idx in unmatched_cols:
            rect = rects[c_idx]
            oid = self.next_id; self.next_id += 1
            self.objects[oid] = rect
            self.centroids[oid] = input_centroids[c_idx]
            self.disappeared[oid] = 0

        return self.objects


# ------------------------------
# Recognition helpers (LBPH via OpenCV contrib)
# ------------------------------

def _has_cv2_face() -> bool:
    return hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')


def _load_images_from_dir(root: str, face_detector: cv2.CascadeClassifier, min_size: Tuple[int, int]):
    images = []
    labels = []
    label_map: Dict[int, str] = {}
    next_label = 0
    for person in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person)
        if not os.path.isdir(person_dir):
            continue
        label_map[next_label] = person
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            if not os.path.isfile(path):
                continue
            if not path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm")):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # if images are not cropped, attempt detect first face
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            roi = gray[y:y+h, x:x+w]
            images.append(roi)
            labels.append(next_label)
        next_label += 1
    return images, labels, label_map


def init_recognizer(args, cascade: cv2.CascadeClassifier, min_size: Tuple[int, int]):
    if not _has_cv2_face():
        if args.recognition_model or args.train_recognition:
            print("Warning: cv2.face not available. Recognition disabled. Install opencv-contrib-python.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_map: Dict[int, str] = {}

    if args.recognition_model and os.path.exists(args.recognition_model):
        try:
            recognizer.read(args.recognition_model)
        except Exception as e:
            print(f"Warning: failed to load recognition model: {e}. Recognition disabled.")
            return None, {}
        return recognizer, label_map

    if args.train_recognition:
        train_dir = args.train_recognition
        if not os.path.isdir(train_dir):
            print(f"Warning: train directory not found: {train_dir}. Recognition disabled.")
            return None, {}
        if np is None:
            print("Warning: numpy not available; cannot train recognizer.")
            return None, {}
        images, labels, label_map = _load_images_from_dir(train_dir, cascade, min_size)
        if not images:
            print("Warning: no training images found. Recognition disabled.")
            return None, {}
        recognizer.train(images, np.array(labels))
        if args.save_recognition_model:
            try:
                recognizer.write(args.save_recognition_model)
                print(f"Saved recognition model to {args.save_recognition_model}")
            except Exception as e:
                print(f"Warning: failed to save recognition model: {e}")
        return recognizer, label_map

    return None, {}


# ------------------------------
# Landmarks and alignment (OpenCV Facemark LBF)
# ------------------------------

def init_facemark(model_path: str):
    if not model_path:
        return None
    if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'createFacemarkLBF'):
        print("Warning: Facemark LBF not available. Install opencv-contrib-python.")
        return None
    fm = cv2.face.createFacemarkLBF()
    try:
        fm.loadModel(model_path)
    except Exception as e:
        print(f"Warning: failed to load landmarks model: {e}")
        return None
    return fm


def align_face(gray_frame, rect, landmarks) -> Optional[np.ndarray]:
    if np is None or landmarks is None:
        return None
    # Use eyes (points around 36-45 for 68-point model) if available; else skip
    pts = landmarks.reshape(-1, 2)
    if pts.shape[0] < 46:
        return None
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))

    x, y, w, h = rect
    center = (x + w // 2, y + h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray_frame, M, (gray_frame.shape[1], gray_frame.shape[0]))
    aligned = rotated[y:y + h, x:x + w]
    return aligned


# ------------------------------
# Main
# ------------------------------

def main():
    args = parse_args()

    cascade_path = args.cascade
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from: {cascade_path}")

    source_value, is_camera = resolve_source(args.source)

    # Optional components
    recognizer, label_map = init_recognizer(args, cascade, tuple(args.min_size))
    facemark = init_facemark(args.landmarks_model) if args.landmarks_model else None
    tracker = CentroidTracker() if args.track else None

    # If image file, process once and exit
    if isinstance(source_value, str) and os.path.isfile(source_value) and any(
        source_value.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ):
        img = cv2.imread(source_value)
        if img is None:
            raise RuntimeError(f"Failed to read image: {source_value}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors,
                                         minSize=tuple(args.min_size))

        # landmarks (if available)
        faces_list = [(x, y, w, h) for (x, y, w, h) in faces]
        landmarks_all = []
        if facemark is not None and len(faces_list) > 0:
            # OpenCV Facemark expects rectangles as (x, y, w, h)
            ok, landmarks = facemark.fit(img, faces_list)
            if ok:
                landmarks_all = [lm for lm in landmarks]

        for idx, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y:y + h, x:x + w]

            # align if requested and landmarks available
            if args.align and facemark is not None and idx < len(landmarks_all):
                aligned = align_face(gray, (x, y, w, h), landmarks_all[idx])
                if aligned is not None:
                    roi_gray = aligned

            # recognition
            label_text = None
            if recognizer is not None and np is not None:
                try:
                    pred_label, confidence = recognizer.predict(roi_gray)
                    name = label_map.get(pred_label, f"ID {pred_label}") if label_map else f"ID {pred_label}"
                    label_text = f"{name} ({confidence:.0f})"
                except Exception:
                    # ignore prediction errors for non-contrib builds
                    pass

            # draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw label if any
            if label_text:
                cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # draw landmarks
            if facemark is not None and idx < len(landmarks_all):
                pts = landmarks_all[idx].reshape(-1, 2)
                for (lx, ly) in pts.astype(int):
                    cv2.circle(img, (int(lx), int(ly)), 1, (255, 0, 0), 1)

        if args.display:
            cv2.imshow("FaceDetect", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        out_path = os.path.splitext(source_value)[0] + "_faces.png"
        cv2.imwrite(out_path, img)
        print(f"Saved image with detections to {out_path}. Faces: {len(faces)}")
        return

    # Else treat as video/camera
    cap = open_capture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.save_video.lower().endswith(".mp4") else "XVID"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(args.save_video, fourcc, args.output_fps, (width, height))
        if not writer.isOpened():
            print(f"Warning: could not open video writer at {args.save_video}. Continuing without saving.")
            writer = None

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=args.scale_factor, minNeighbors=args.min_neighbors,
                                         minSize=tuple(args.min_size))
        rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        # recognition + landmarks + alignment
        landmarks_all = []
        if facemark is not None and len(rects) > 0:
            ok, landmarks = facemark.fit(frame, rects)
            if ok:
                landmarks_all = [lm for lm in landmarks]

        for idx, (x, y, w, h) in enumerate(rects):
            roi_gray = gray[y:y + h, x:x + w]

            if args.align and facemark is not None and idx < len(landmarks_all):
                aligned = align_face(gray, (x, y, w, h), landmarks_all[idx])
                if aligned is not None:
                    roi_gray = aligned

            # recognition
            label_text = None
            if recognizer is not None and np is not None:
                try:
                    pred_label, confidence = recognizer.predict(roi_gray)
                    name = label_map.get(pred_label, f"ID {pred_label}") if label_map else f"ID {pred_label}"
                    label_text = f"{name} ({confidence:.0f})"
                except Exception:
                    pass

            # draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if label_text:
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if facemark is not None and idx < len(landmarks_all):
                pts = landmarks_all[idx].reshape(-1, 2)
                for (lx, ly) in pts.astype(int):
                    cv2.circle(frame, (int(lx), int(ly)), 1, (255, 0, 0), 1)

        # tracking: assign IDs
        if tracker is not None:
            objects = tracker.update(rects)
            for oid, (x, y, w, h) in objects.items():
                cv2.putText(frame, f"T{oid}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # FPS calculation
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        draw_fps(frame, fps)

        if args.display:
            cv2.imshow("FaceDetect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
