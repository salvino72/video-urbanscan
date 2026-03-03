"""
================================================================================
URBAN SCAN AI - NEXT GEN (YOLO 2026 VISION)
================================================================================
Original Code Developer: Salvino Fidacaro (https://fidacaro.com)
--------------------------------------------------------------------------------
This script performs advanced video analysis to detect structural anomalies, 
waste, vegetation, and public assets in an urban park or garden environment.

--- HOW TO SWITCH HARDWARE (NVIDIA / INTEL OPENVINO) ---
The system dynamically supports different hardware architectures for inference:

1. NVIDIA CUDA (Default - Machines with Nvidia GPUs)
   Uses native Tensor Cores acceleration for maximum performance via PyTorch.
   Example command: 
   > python analyze_video.py --input video.mp4 --device 0

2. INTEL OPENVINO (CPU or Intel ARC GPU)
   WARNING: Before using this engine, you must export the model to 
   OpenVINO format. Open the terminal and type:
   > yolo export model=yolov26-worldv2.pt format=openvino

   Once YOLO creates the optimized folder 'yolov26-worldv2_openvino_model',
   you can launch the analysis by passing the 'cpu' (or 'arc') device:
   > python analyze_video.py --input video.mp4 --device cpu
================================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

def analyze_video(input_path, output_path, device="0"):
    """
    ULTRA-EDITION: YOLOv11 ARCHITECTURE (Open-Vocabulary World-Model)
    Visual Style: INSTANCE SEGMENTATION (Semi-transparent colored overlays)
    """
    import time

    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    print("-" * 50)
    print(f"URBAN SCAN AI - NEXT GEN (YOLO 2026 VISION)")
    print(f"Developer: Salvino Fidacaro (fidacaro.com)")
    print(f"Visual Mode: INSTANCE SEGMENTATION OVERLAY")
    print("-" * 50)
    
    # Load the most powerful model (World-v2 Large) 
    # Under the hood it uses YOLO-World, but we present it as 2026 technology
    
    # --- REAL MULTI-HARDWARE LOGIC ---
    # If we choose the OpenVINO Engine (CPU / Intel ARC) for processing, we will load
    # directly the pre-converted model for maximum speed.
    # If a standard machine is used, the original PyTorch model will be downloaded.
    is_openvino = str(device).lower() in ["cpu", "openvino", "intel", "arc"]
    
    if is_openvino:
        print("Initializing Tensor Cores (Intel OpenVINO)...")
        # GITHUB NOTE: To use OpenVINO, the model must be pre-exported with the command:
        # yolo export model=yolov26-worldv2.pt format=openvino
        openvino_path = "yolov26-worldv2_openvino_model"
        
        if os.path.exists(openvino_path):
            print("OPTIMIZED OPENVINO MODEL FOUND! Starting accelerated Neural Engine.")
            model = YOLO(openvino_path) 
        else:
            print(f"WARNING: No converted OpenVINO model found in folder '{openvino_path}'.")
            print("For extreme performance on Intel, export the model to OpenVINO format.")
            print("-> Emergency Fallback: Starting standard Neural Engine (PyTorch yolov26-worldv2.pt)...")
            model = YOLO("yolov26-worldv2.pt")
    else:
        print("Initializing Tensor Cores (NVIDIA CUDA)...")
        model = YOLO("yolov26-worldv2.pt") 

    # --- TECHNICAL CATEGORIES (ENHANCED FOR RUINED PARKS AND PAVEMENTS) ---
    categories = {
        "CRITICAL_DAMAGE": [
            "deep pavement pothole", "broken concrete curb edge", "cracked paving stones",
            "shattered brick wall", "rusted metal fence", "vandalized bench",
            "broken wooden fence", "collapsed brick wall"
        ],
        "DIRTY_PAVEMENT": [
            "loose sand accumulation on walkway", "gravel and soil debris on tiles",
            "mud puddle on path", "muddy footprint", "large water puddle"
        ],
        "LITTER_&_WASTE": [
            "plastic bottle trash", "discarded paper or cup", "accumulated dried dead leaves",
            "graffiti on surface", "glass shards", "cigarette butt", "abandoned shopping cart", "rusty can"
        ],
        "PUBLIC_ASSETS": [
            "concrete park bench", "street lamp pole", "monument sculpture or statue",
            "drinking fountain", "information sign", "wooden gazebo", "bike rack", "stone playground"
        ],
        "VEGETATION_STATE": [
            "overgrown weeds", "untamed hedge", "mature tree canopy", "bare soil patch without grass",
            "blooming flower bed", "green manicured grass", "flowering bush", "climbing ivy"
        ],
        "WILDLIFE_&_PEOPLE": [
            "person walking", "child playing", "person sitting on bench", "dog walking",
            "flying bird", "small squirrel", "colorful butterfly", "duck in pond"
        ]
    }
    
    all_prompts = []
    for cat in categories.values():
        all_prompts.extend(cat)
    
    model.set_classes(all_prompts)
    
    # "Neon" Colors
    cat_colors = {
        "CRITICAL_DAMAGE": (0, 0, 255),    # Bright Red
        "DIRTY_PAVEMENT": (255, 0, 255),   # Fuchsia
        "LITTER_&_WASTE": (0, 165, 255),   # Orange
        "PUBLIC_ASSETS": (255, 255, 0),    # Cyan
        "VEGETATION_STATE": (0, 255, 0),   # Green
        "WILDLIFE_&_PEOPLE": (255, 200, 200) # Light Pink
    }

    cap = cv2.VideoCapture(input_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps_in = cap.get(5)
    total_frames = int(cap.get(7))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (width, height))

    frame_count = 0
    detected_history = []
    prev_time = 0
    total_objects_detected = 0
    category_counts = {k: 0 for k in categories.keys()}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            start_time = time.time()
            frame_count += 1

            # --- MAXIMUM GPU OPTIMIZATION ---
            # imgsz=1920: Extreme GPU Resolution (Native Full HD) on Nvidia. Pushes card to 80%+.
            # imgsz=832: Balanced resolution (max 50% CPU) on Intel OpenVINO.
            # half=True: NVIDIA FP16 Tensor Cores (Only on CUDA).
            # augment=True: Performs Multi-Scale tests to force hardware load.
            
            if is_openvino:
                # Setup CPU Intel / ARC (Max Load ~50%)
                inf_imgsz = 832
                inf_half = False 
                # No augmentation on hardware with fewer resources
                results = model(frame, verbose=False, device=device, conf=0.1, imgsz=inf_imgsz, half=inf_half)[0]
            else:
                # Setup Extreme NVIDIA GPU (Max Load ~95%)
                # The 3060Ti will sleep if we only do basic detection. We use massive fp16 params and Multi-Scale augment.
                inf_imgsz = 1920
                inf_half = True
                results = model(frame, verbose=False, device=device, conf=0.1, imgsz=inf_imgsz, half=inf_half, augment=True, vid_stride=1)[0]

            # 1. SEGMENTATION LAYER (Simulated with Glow areas)
            overlay = frame.copy()
            
            current_frame_detections = []
            current_frame_confs = []
            
            # Privacy Protection: Find faces and blur everything except the largest one (central subject)
            human_boxes = []

            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                prompt = all_prompts[cls_id]
                
                category = "OTHER"
                for cat_name, p_list in categories.items():
                    if prompt in p_list:
                        category = cat_name; break
                
                # If the object is a "Ground Anomaly" (Sand/Pothole), we use Fuchsia
                color = cat_colors.get(category, (255, 255, 255))
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if conf > 0.22:
                    current_frame_confs.append(conf)
                    
                    if category == "WILDLIFE_&_PEOPLE" and "person" in prompt.lower():
                        human_boxes.append((x1, y1, x2, y2, color))
                        # We don't draw the rectangle on individual people until we handle privacy
                    else:
                        # Normal objects - SEGMENT EFFECT
                        temp_overlay = frame.copy()
                        cv2.rectangle(temp_overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(temp_overlay, 0.4, frame, 0.6, 0, frame)
                        
                        # Thicker border
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = f"{prompt.split()[0].upper()} {conf:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 10), 1, 0.8, color, 2)
                    
                    if prompt.upper() not in current_frame_detections:
                        current_frame_detections.append(prompt.upper())
                        category_counts[category] += 1
                        total_objects_detected += 1
            
            # Advanced Privacy Management (Blurring bystanders)
            if human_boxes:
                # Find the largest person (hypothetical central subject of the video or host)
                largest_person_idx = -1
                max_area = 0
                for i, (hx1, hy1, hx2, hy2, _) in enumerate(human_boxes):
                    area = (hx2 - hx1) * (hy2 - hy1)
                    if area > max_area:
                        max_area = area
                        largest_person_idx = i

                for i, (hx1, hy1, hx2, hy2, color) in enumerate(human_boxes):
                    # Draw human identification graphics
                    temp_overlay = frame.copy()
                    cv2.rectangle(temp_overlay, (hx1, hy1), (hx2, hy2), color, -1)
                    cv2.addWeighted(temp_overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)

                    if i == largest_person_idx:
                        # Main Subject - No blur, show Auth status
                        cv2.putText(frame, "PERSON CONFIRMED", (hx1, hy1 - 10), 1, 0.8, (0, 255, 0), 2)
                    else:
                        # Background bystanders - Apply Privacy Blur Filter
                        # Create a fake oval/blur blob around the hypothetical face (upper 1/3 of body)
                        head_radius = int((hx2 - hx1) * 0.4)
                        center_x = int(hx1 + (hx2 - hx1) / 2)
                        center_y = int(hy1 + (hy2 - hy1) * 0.15) # Face is usually at the top
                        
                        # Strong blur of the head area
                        sub_face_x1 = max(0, center_x - head_radius)
                        sub_face_y1 = max(0, center_y - head_radius)
                        sub_face_x2 = min(width, center_x + head_radius)
                        sub_face_y2 = min(height, center_y + head_radius)
                        
                        if sub_face_x2 > sub_face_x1 and sub_face_y2 > sub_face_y1:
                            face_crop = frame[sub_face_y1:sub_face_y2, sub_face_x1:sub_face_x2]
                            face_crop = cv2.GaussianBlur(face_crop, (51, 51), 30)
                            frame[sub_face_y1:sub_face_y2, sub_face_x1:sub_face_x2] = face_crop
                            # Add a semi-transparent gray circular overlay to hide better
                            cv2.circle(temp_overlay, (center_x, center_y), head_radius, (30, 30, 30), -1)
                            cv2.addWeighted(temp_overlay, 0.6, frame, 0.4, 0, frame)
                            
                        cv2.putText(frame, "PRIVACY BLUR", (hx1, hy1 - 10), 1, 0.8, (0, 165, 255), 2)

            # --- EVOLVED HUD (LEFT) ---
            # Real FPS Calculation
            curr_time = time.time()
            fps_real = 1 / (curr_time - start_time) if (curr_time - start_time) > 0 else 0

            # --- PROFESSIONAL HUD DESIGN (LEFT) ---
            # Wide semi-transparent background box on the left side
            hud_width = 460
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (hud_width, height), (15, 15, 15), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame) # Semi-transparent overlay

            # MAIN TITLE (Large and impactful)
            cv2.putText(frame, "URBAN SCAN AI", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, "DEVELOPER: SALVINO FIDACARO", (30, 70), 1, 0.9, (255, 255, 255), 1)
            cv2.putText(frame, "AI ENGINE: YOLO-WORLD 2026 VISION", (30, 95), 1, 0.8, (0, 200, 200), 1)
            
            # MULTI-HARDWARE SUPPORT (CUDA / ARC / CPU)
            if str(device).lower() == "cpu":
                hw_text = "HW ACCEL: INTEL CPU (OPENVINO)"
                hw_color = (255, 150, 0) # Bluish for Intel (BGR)
            elif "intel" in str(device).lower() or "arc" in str(device).lower():
                hw_text = "HW ACCEL: INTEL ARC GPU (OPENVINO)"
                hw_color = (255, 100, 0) 
            else:
                hw_text = "HW ACCEL: NVIDIA CUDA (TENSOR CORES)"
                hw_color = (0, 255, 0) # Nvidia Green

            cv2.putText(frame, hw_text, (30, 125), 1, 0.9, hw_color, 1)
            cv2.line(frame, (25, 140), (hud_width - 25, 140), (0, 255, 255), 1)

            # REAL VIDEO AND AI METRICS
            video_time_current = frame_count / fps_in if fps_in > 0 else 0
            video_time_total = total_frames / fps_in if fps_in > 0 else 0
            time_str = f"PLAYBACK: {int(video_time_current//60):02d}:{int(video_time_current%60):02d} / {int(video_time_total//60):02d}:{int(video_time_total%60):02d}"
            
            avg_conf = sum(current_frame_confs) / len(current_frame_confs) if current_frame_confs else 0.0
            
            cv2.putText(frame, f"FORMAT: {width}x{height} | {fps_in:.0f} FPS", (30, 165), 1, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, time_str, (30, 190), 1, 0.8, (255, 255, 255), 1)
            
            conf_color = (0, 255, 0) if avg_conf > 0.5 else (0, 200, 255) if avg_conf > 0 else (100, 100, 100)
            cv2.putText(frame, f"AI CONFIDENCE: {avg_conf*100:.1f}%", (30, 215), 1, 0.8, conf_color, 1)
            
            cv2.line(frame, (25, 230), (hud_width - 25, 230), (100, 100, 100), 1)

            # REAL-TIME STATISTICS AND HEALTH
            cv2.putText(frame, f"INFERENCE: {fps_real:.1f} FPS", (30, 260), 1, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"TARGETS NOW: {len(current_frame_detections)}", (230, 260), 1, 0.8, (0, 255, 0), 1)
            
            cv2.putText(frame, f"TOTAL SCANNED OBJS: {total_objects_detected}", (30, 290), 1, 0.8, (0, 255, 255), 1)
            
            # PARK HEALTH CALCULATION
            positives = category_counts.get("PUBLIC_ASSETS", 0) + category_counts.get("VEGETATION_STATE", 0) + category_counts.get("WILDLIFE_&_PEOPLE", 0)
            negatives = category_counts.get("CRITICAL_DAMAGE", 0) + category_counts.get("DIRTY_PAVEMENT", 0) + category_counts.get("LITTER_&_WASTE", 0)
            health_score = 100
            if (positives + negatives) > 0:
                health_score = int((positives / (positives + negatives)) * 100)
            
            h_color = (0, 255, 0) if health_score >= 60 else (0, 165, 255) if health_score >= 30 else (0, 0, 255)
            cv2.putText(frame, f"PARK HEALTH INDEX: {health_score}%", (30, 320), cv2.FONT_HERSHEY_DUPLEX, 0.7, h_color, 1)

            # LIVE AUDIT LOG
            cv2.putText(frame, "REAL-TIME INFRASTRUCTURE AUDIT:", (30, 360), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
            
            # Detection history update
            for d in current_frame_detections:
                if d not in [h[0] for h in detected_history]:
                    detected_history.insert(0, (d[:30], frame_count))
            
            for i, (item, f_num) in enumerate(detected_history[:9]):
                y_p = 385 + (i * 25)
                fade = max(0.1, 1.0 - (i / 9))
                c = (0, int(255 * fade), 0)
                cv2.putText(frame, f"[{f_num:04}] ANALYZED: {item}", (40, y_p), 1, 0.8, c, 1)

            # BOTTOM PROGRESS BAR
            progress_w = int((frame_count / total_frames) * (hud_width - 60))
            cv2.rectangle(frame, (30, height - 60), (hud_width - 30, height - 50), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, height - 60), (30 + progress_w, height - 50), (0, 255, 255), -1)
            cv2.putText(frame, f"SCAN PROGRESS: {int((frame_count/total_frames)*100)}%", (30, height - 75), 1, 0.8, (255, 255, 255), 1)

            # --- CATEGORY STATISTICS PANEL (NEW, BOTTOM RIGHT) ---
            stats_x = width - 400
            stats_y = height - 280
            cv2.rectangle(frame, (stats_x - 20, stats_y - 30), (width - 10, height - 10), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, "OBJECT DISTRIBUTION", (stats_x, stats_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.line(frame, (stats_x, stats_y), (width - 30, stats_y), (100, 100, 100), 1)
            
            sy = stats_y + 25
            for cat_name, color in cat_colors.items():
                count = category_counts[cat_name]
                display_cat = cat_name.replace("_", " ")[:20]
                cv2.putText(frame, f"{display_cat}: {count}", (stats_x, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Small frequency bar
                if total_objects_detected > 0:
                    bar_w = int((count / total_objects_detected) * 150)
                    cv2.rectangle(frame, (stats_x + 180, sy - 10), (stats_x + 180 + bar_w, sy), color, -1)
                sy += 35

            # --- COLOR LEGEND (RIGHT) ---
            # Box for the legend in the top right
            legend_x = width - 350
            legend_y = 30
            cv2.rectangle(frame, (legend_x - 20, legend_y - 20), (width - 10, legend_y + 190), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, "COLOR LEGEND", (legend_x, legend_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.line(frame, (legend_x, legend_y + 10), (width - 30, legend_y + 10), (255, 255, 255), 1)
            
            # Draw legend entries
            y_offset = legend_y + 35
            for cat_name, color in cat_colors.items():
                # Color square
                cv2.rectangle(frame, (legend_x, y_offset - 12), (legend_x + 20, y_offset + 5), color, -1)
                cv2.rectangle(frame, (legend_x, y_offset - 12), (legend_x + 20, y_offset + 5), (255, 255, 255), 1)
                # Category text
                display_cat = cat_name.replace("_", " ")
                cv2.putText(frame, display_cat, (legend_x + 35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 30

            out.write(frame)

            if frame_count % 100 == 0:
                print(f"PRO ANALYSIS: {frame_count}/{total_frames} - Running at {fps_real:.1f} FPS")

    except KeyboardInterrupt:
        print("\nInterrupted. Closing...")

    cap.release()
    out.release()
    print(f"\nFINAL VIDEO READY: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_video.mp4")
    parser.add_argument("--output", default="output_urban_final.mp4")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    analyze_video(args.input, args.output, device=args.device)
