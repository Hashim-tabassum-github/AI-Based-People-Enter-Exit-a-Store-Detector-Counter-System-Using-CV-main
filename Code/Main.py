# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import csv
from datetime import datetime
from pathlib import Path
import time


track_history = defaultdict(list)
track_states = defaultdict(lambda: {
    'inside': False, 
    'previous_inside': False,
    'last_count_frame': -1,
    'cooldown_frames': 0
})

current_region = None
counting_regions = [
    {
        "name": "Ultralytics YOLO Polygon Region",
        "polygon": Polygon([(1205, 253), (1162, 678), (1410, 811), (1682, 713)]),   # Polygon points
        "enter_count": 0,
        "exit_count": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    }
]


def get_movement_direction(track_points):
    """Flexible direction analysis for both entering and exiting."""
    if len(track_points) < 3:
        return None

    num_points = min(5, len(track_points) // 2)
    if num_points < 2:
        num_points = len(track_points) // 2 if len(track_points) >= 4 else 1

    start_points = track_points[:num_points]
    end_points = track_points[-num_points:]

    start_avg = (
        sum(p[0] for p in start_points) / len(start_points),
        sum(p[1] for p in start_points) / len(start_points)
    )
    end_avg = (
        sum(p[0] for p in end_points) / len(end_points),
        sum(p[1] for p in end_points) / len(end_points)
    )

    dx = end_avg[0] - start_avg[0]
    dy = end_avg[1] - start_avg[1]

    min_movement = 10
    if abs(dx) < min_movement and abs(dy) < min_movement:
        return None

    horizontal_movement = abs(dx)
    vertical_movement = abs(dy)

    if horizontal_movement >= vertical_movement * 0.7:
        return 'entering' if dx > 0 else 'exiting'
    else:
        return 'entering' if dy < 0 else 'exiting'


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Handle mouse events for region manipulation in the video frame."""
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights: str = "yolo11l.pt",
    source: str = None,
    device: str = "cuda",
    view_img: bool = False,
    save_img: bool = False,
    exist_ok: bool = False,
    classes: list[int] = 0,
    line_thickness: int = 1,
    track_thickness: int = 1,
    region_thickness: int = 1,
    skip_frames: int = 5,
    no_regions_lines: bool = False,
) -> None:
    """Main loop for YOLO people counter with CSV logging."""
    vid_frame_count = 0
    process_frame_count = 0

    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    names = model.names

    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = increment_path(Path(r"D:\Yolo_obj_detection\ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"), fourcc, fps, (frame_width, frame_height))

    output_path = str(save_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"{timestamp}_{Path(source).stem}.csv"
    csv_path = save_dir / csv_filename

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "entering", "exiting", "total", "fps"])

    print(f"\n📹 Saving processed video to: {output_path}")
    print("🚀 Processing started with people detection and counting enabled...")
    print("📊 Watch the top-right corner for live statistics")
    print("👁️ Region lines are hidden for clean output")

    last_results = None
    start_time = time.time()
    frame_count = 0
    last_log_second = -1

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
        frame_count += 1
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # === Periodic CSV Logging (once per second) ===
        log_interval = 10  # change to 20 if you want every 20 seconds
        current_second = int(elapsed_time)
        if current_second % log_interval == 0 and current_second != last_log_second:
            entering = counting_regions[0]["enter_count"]
            exiting = counting_regions[0]["exit_count"]
            total = entering + exiting
            csv_writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                entering,
                exiting,
                total,
                f"{current_fps:.1f}"
            ])
            last_log_second = current_second


        process_this_frame = (vid_frame_count % (skip_frames + 1) == 0)

        if process_this_frame:
            process_frame_count += 1
            results = model.track(frame, persist=True, classes=classes)
            last_results = results
        else:
            results = last_results

        if results and results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                track = track_history[track_id]

                if process_this_frame:
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)

                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                if process_this_frame:
                    for region in counting_regions:
                        current_inside = region["polygon"].contains(Point((bbox_center[0], bbox_center[1])))

                        prev_inside = track_states[track_id]['previous_inside']
                        cooldown = track_states[track_id]['cooldown_frames']

                        if cooldown > 0:
                            track_states[track_id]['cooldown_frames'] -= 1

                        if (current_inside != prev_inside and 
                            cooldown == 0 and 
                            len(track) >= 5 and  
                            abs(process_frame_count - track_states[track_id]['last_count_frame']) > 10):

                            direction = get_movement_direction(track[-6:] if len(track) >= 6 else track)

                            if direction:
                                if not prev_inside and current_inside:
                                    if direction == 'entering':
                                        region["enter_count"] += 1
                                    else:
                                        region["exit_count"] += 1
                                    track_states[track_id]['last_count_frame'] = process_frame_count
                                    track_states[track_id]['cooldown_frames'] = 20
                                elif prev_inside and not current_inside:
                                    if direction == 'exiting':
                                        region["exit_count"] += 1
                                    else:
                                        region["enter_count"] += 1
                                    track_states[track_id]['last_count_frame'] = process_frame_count
                                    track_states[track_id]['cooldown_frames'] = 20

                        track_states[track_id]['previous_inside'] = current_inside

        # === Draw counter boxes (top-right) ===
        entering = counting_regions[0]["enter_count"]
        exiting = counting_regions[0]["exit_count"]
        total = entering + exiting
        current_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

        box_width = int(frame_width * 0.18)
        box_height = int(frame_height * 0.07)
        margin = int(frame_height * 0.015)
        font_scale = 0.9 * (frame_width / 1920.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = 2

        counters = [
            ("ENTERING", entering, (0, 255, 0)),
            ("EXITING", exiting, (0, 0, 255)),
            ("TOTAL", total, (255, 0, 0)),
            ("FPS", f"{current_fps:.1f}", (0, 200, 255)),
        ]

        x1 = frame_width - box_width - 20
        y1 = 20
        for label, value, color in counters:
            cv2.rectangle(frame, (x1, y1), (x1 + box_width, y1 + box_height), (255, 255, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x1 + 8, y1 + box_height), color, -1)
            cv2.putText(frame, label, (x1 + 15, y1 + int(box_height * 0.45)),
                        font, font_scale, (128, 128, 128), text_thickness, cv2.LINE_AA)
            text_color = (255, 0, 0) if label != "FPS" else (0, 140, 255)
            text = str(value)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale * 1.2, text_thickness + 1)
            cv2.putText(frame, text, (x1 + box_width - tw - 15, y1 + int(box_height * 0.7)),
                        font, font_scale * 1.2, text_color, text_thickness + 1, cv2.LINE_AA)
            y1 += box_height + margin

        if not no_regions_lines:
            for region in counting_regions:
                polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                cv2.polylines(frame, [polygon_coordinates], isClosed=True, color=region["region_color"], thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLO Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLO Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLO Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # === Final statistics ===
    avg_fps = frame_count / (time.time() - start_time)
    csv_file.close()

    print("\n" + "="*50)
    print("📊 FINAL STATISTICS - PEOPLE COUNTING")
    print("="*50)
    print(f"👥 People Entering: {counting_regions[0]['enter_count']}")
    print(f"🚶 People Exiting: {counting_regions[0]['exit_count']}")
    print(f"📈 Total People Count: {counting_regions[0]['enter_count'] + counting_regions[0]['exit_count']}")
    print(f"🎬 Total Frames Processed: {frame_count}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"✅ CSV log saved at: {csv_path}")

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11l.pt", help="initial weights path")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", default=True, action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", default=0, type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")
    parser.add_argument("--skip-frames", type=int, default=2, help="Skip frames for faster processing (process every nth frame)")
    parser.add_argument("--no-regions-lines", action="store_true", help="Hide region lines for clean display")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
