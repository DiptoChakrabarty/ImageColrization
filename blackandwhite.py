import cv2
import os
import numpy as np
path="./colored"
upload="./black"

colored=list(os.path.join(path,f) for f in os.listdir(path))
count=0
for images in  colored:
    count+=1
    photo=cv2.imread(images)
    resized=cv2.resize(photo,(300,300))
    cv2.imshow("lakes",photo)
    cv2.waitKey()
    cv2.destroyAllWindows()

    black=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(upload+"image{}".format(count),black)