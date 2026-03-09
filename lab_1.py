import cv2
from ultralytics import YOLO
import os
import time

# Load the YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

def run_image_detection(image_path):
    """Run object detection on an image"""
    print(f"\n🔍 Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File {image_path} not found!")
        return
    
    results = model(image_path, conf=0.25)
    annotated_image = results[0].plot()
    num_objects = len(results[0].boxes)
    print(f"📊 Detected {num_objects} objects in the image")
    
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ Image saved as: {output_path}")
    
    cv2.imshow(f'YOLOv8 Detection - {num_objects} objects found', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ Image detection completed!")

def run_video_detection(video_path):
    """Run object detection on a video"""
    print(f"\n🎥 Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File {video_path} not found!")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    total_objects = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.track(frame, persist=True, conf=0.25)
        annotated_frame = results[0].plot()
        
        if results[0].boxes is not None:
            frame_objects = len(results[0].boxes)
            total_objects += frame_objects
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"✅ Video saved as: output_video.mp4")
    print(f"Total frames processed: {frame_count}")
    print(f"Average objects per frame: {total_objects/frame_count:.1f}")

def run_webcam_detection():
    """Run real-time object detection using webcam - SLOWER for better recognition"""
    print("\n" + "=" * 50)
    print("📹 SLOW MODE - OPTIMIZED FOR MAXIMUM RECOGNITION")
    print("=" * 50)
    print("Press 'q' to quit recording")
    print("\n💡 TIPS FOR BEST DETECTION:")
    print("• Hold objects STEADY for 2-3 seconds")
    print("• Ensure GOOD LIGHTING")
    print("• Show objects ONE AT A TIME first")
    print("• Then group them together")
    print("-" * 50)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📷 Webcam: {width}x{height} (SLOW MODE = BETTER RECOGNITION)")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('webcam_recording.mp4', fourcc, 15.0, (width, height))
    
    print("\n✅ Recording started! Take your time showing objects...")
    print("-" * 40)
    
    frame_count = 0
    object_history = []
    start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        results = model.track(frame, persist=True, conf=0.2, iou=0.3, verbose=False)
        annotated_frame = results[0].plot()
        
        if results[0].boxes is not None:
            num_objects = len(results[0].boxes)
            object_history.append(num_objects)
            
            class_ids = results[0].boxes.cls.cpu().numpy()
            conf_scores = results[0].boxes.conf.cpu().numpy()
            class_names = [model.names[int(id)] for id in class_ids]
            detection_info = list(zip(class_names, conf_scores))
        else:
            num_objects = 0
            detection_info = []
        
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add information on frame
        cv2.putText(annotated_frame, f"SLOW MODE - Better Recognition", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show detected objects
        y_pos = 150
        cv2.putText(annotated_frame, "Detected:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for i, (obj, conf) in enumerate(detection_info[:8]):
            y_pos += 25
            cv2.putText(annotated_frame, f"• {obj} ({conf*100:.0f}%)", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('SLOW MODE: YOLOv8 Detection - Hold objects steady', annotated_frame)
        out.write(annotated_frame)
        
        time.sleep(0.05)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    total_time = time.time() - start_time
    avg_objects = sum(object_history) / len(object_history) if object_history else 0
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("📊 RECORDING STATISTICS")
    print("=" * 50)
    print(f"✅ Webcam recording saved: webcam_recording.mp4")
    print(f"📹 Total frames recorded: {frame_count}")
    print(f"⏱️  Duration: {total_time:.1f} seconds")
    print(f"📦 Avg objects detected: {avg_objects:.1f}")
    print("=" * 50)

# ============================================
# MAIN EXECUTION
# ============================================
print("=" * 50)
print("YOLOv8 Object Detection Lab")
print("=" * 50)

# UNCOMMENT THE TEST YOU WANT TO RUN:

# Image Detection
# source = "image.jpg"
# run_image_detection(source)

# Video Detection
# source = "video.mp4"
# run_video_detection(source)

# Webcam Detection (CURRENTLY ACTIVE)
print("\n📹 Running WEBCAM DETECTION...")
print("🐢 SLOW MODE ENABLED")
run_webcam_detection()

print("\n" + "=" * 50)
print("✅ Script execution completed!")
print("=" * 50)