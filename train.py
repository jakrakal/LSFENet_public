import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/v11-RGBT/yolo11-RGBT-midfusion-P3-LSFENet.yaml') 
    model.train(data='./ultralytics/cfg/datasets/LLVIP.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=5,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/LLVIP',
                name='LLVIP-yolo11-RGBT-midfusion-P3-LSFENet-e300-SGD',
                )