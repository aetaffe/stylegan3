from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n-seg.pt')
    model.info()
    model.train(data='/media/alex/1TBSSD/research_gans/Generated_Data/Generated_Images_512x512_yolo/synthetic-FLIm-512x512.yaml', epochs=100, imgsz=512)
