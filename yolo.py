from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data = "polyp.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 32,
    device = 0,

)

