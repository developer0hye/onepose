# onepose

Human **pose** estimation based on [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) within **one line** 

```bash
pip install git+https://github.com/developer0hye/onepose.git
```

```python
import cv2
import onepose
img = cv2.imread("sample.png")
keypoints = onepose.create_model().to("cuda")(img)
```

![sample_vitpose_h_simple_coco_output](./onepose/assets/sample_vitpose_h_simple_coco_output.png)

One Piece's Luffy pose predicted by onepose

# Examples

## Plot keypoints on an image
```python
import cv2
import onepose

if __name__ == '__main__':
    img = cv2.imread('sample.png')
    model = onepose.create_model()

    keypoints = model(img)
    num_keypoints = len(keypoints['points'])
    
    for i in range(num_keypoints):
        print(f"Point {i} (x, y)  : {keypoints['points'][i]} confidence: {keypoints['confidence'][i]}")
        
        if keypoints['confidence'][i] < 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        
        cv2.circle(img, (int(keypoints['points'][i][0]), int(keypoints['points'][i][1])), 5, color, -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
```

Notice that occluded key points are plotted in red. You can discard these points using confidence score.

![occluded_sample_vitpose_h_simple_coco_output](./onepose/assets/occluded_sample_vitpose_h_simple_coco_output.png)

## Multiple-person pose estimation with YOLOv8
```bash
pip install ultralytics
```
```python
import cv2
from ultralytics import YOLO
import onepose

detection_model = YOLO("yolov8m.pt")
pose_estimiation_model = onepose.create_model()

img = cv2.imread("multiple_person_sample.png")
draw_img = img.copy()

results = detection_model(img)[0]
boxes = results.boxes.xyxy
clses = results.boxes.cls
probs = results.boxes.conf

for cls, box, prob in zip(clses, boxes, probs):
    if cls != 0:
        continue

    x1, y1, x2, y2 = box
    # crop image
    person_img = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    keypoints = pose_estimiation_model(person_img)
    num_keypoints = len(keypoints['points'])
    
    for i in range(num_keypoints):
        print(f"Point {i} (x, y)  : {keypoints['points'][i]} confidence: {keypoints['confidence'][i]}")
        
        if keypoints['confidence'][i] < 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        
        cv2.circle(draw_img, (int(keypoints['points'][i][0] + x1), int(keypoints['points'][i][1] + y1)), 3, color, -1)

cv2.imshow("draw_img", draw_img)
cv2.waitKey(0)
```
![multiple_person_sample_vitpose_h_simple_coco_output](./onepose/assets/multiple_person_sample_vitpose_h_simple_coco_output.png)


## Print supported models
```python
import onepose
print(onepose.list_models()) # ['ViTPose_base_simple_coco', 'ViTPose_large_simple_coco', 'ViTPose_huge_simple_coco', ...]
```

# References

[ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

[jaehyunnn/ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch)

[JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)
