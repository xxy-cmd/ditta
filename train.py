from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo111.yaml")
    model.load("yolo11m.pt")
    model.train(
        data=r"data.yaml",
        epochs=300,
        device="cpu",  # CPU训练
        batch=2,
        amp=False,
        workers=0,
        imgsz=640,
        pretrained=True,
    )
