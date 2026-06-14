import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    'output.avi',
    fourcc,
    20.0,
    (640, 480)
)

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    # Perform object detection
    results = model.track(frame, persist=True)
    count = 0

    for box in results[0].boxes:
        count += 1
    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()
    cv2.putText(
    annotated_frame,
    f"Objects Count: {count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

    # Display output
    out.write(annotated_frame)
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()