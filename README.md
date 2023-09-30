# onepose

Human **pose** estimation within **one line**

```bash
pip install onepose
```

```python
import cv2
import onepose
img = cv2.imread("sample.png")
keypoints = onepose.create_model().to("cuda")(img)
```

![output](https://github.com/developer0hye/onepose/assets/35001605/373de9d0-a2ea-4b34-8b6c-156e5de71377)

One Piece's Luffy pose predicted by onepose

# Examples

## Plot keypoints on an image
```python
import cv2
import onepose

if __name__ == '__main__':
    img = cv2.imread('sample.png')
    model = onepose.create_model()
    out = model(img)
    num_keypoints = len(out['points'])
    
    for i in range(num_keypoints):
        print(f"Point {i} (x, y)  : {out['points'][i]} confidence: {out['confidence'][i]}")
        
        if out['confidence'][i] < 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        
        cv2.circle(img, (int(out['points'][i][0]), int(out['points'][i][1])), 5, color, -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
```
### Results

Notice that occluded key points are plotted in red. You can discard these points using confidence score.

![output](https://github.com/developer0hye/onepose/assets/35001605/efad3e3f-7ab7-4521-bec1-d5cb2f9007a2)


# References

[ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

[jaehyunnn/ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch)

[JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)
