import argparse
import os
import time
import cv2  # import openCV


def parse_args():
    parser = argparse.ArgumentParser(description="Face detection using OpenCV Haar cascades")
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
    return parser.parse_args()


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


def main():
    args = parse_args()

    cascade_path = args.cascade
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade from: {cascade_path}")

    source_value, is_camera = resolve_source(args.source)

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
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
        

    

 

