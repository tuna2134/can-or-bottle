from onnxruntime import InferenceSession
import numpy as np
from PIL import Image


sess = InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

img = Image.open("./coke.webp")
img = img.resize((224, 224))
img = img.convert("RGB")
img = np.array(img, dtype=np.float32)
img = np.transpose(img, (2, 0, 1))
img = img / 255.0
img = np.expand_dims(img, 0)

result = sess.run(
    [output_name],
    {
        input_name: img,
    },
)

if result[0][0].argmax() == 0:
    print("缶ジュース")
else:
    print("ペットボトル")
print(result[0][0])
