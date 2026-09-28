import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('.../weights/best.pt')
    model.val(data='./ultralytics/cfg/datasets/LLVIP.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBRGB6C",
              channels=6,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val/',
              name='....',
              )