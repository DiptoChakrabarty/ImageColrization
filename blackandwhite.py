import cv2
import os
import numpy as np
path="./colored"

colored=list(os.path.join(path,f) for f in os.listdir(path))

print(colored)