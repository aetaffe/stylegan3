from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/segment/train3/weights/best.pt')
    metrics = model.val()
    print(f'mAP 50 - 95: {metrics.box.map}')
    print(f'mAP 50: {metrics.box.map50}')
    print(f'mAP 75: {metrics.box.map75}')
    print(f'mAP categories: {metrics.box.maps}')
