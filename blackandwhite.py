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
    resized=cv2.resize(photo,(400,350))
    cv2.imwrite(images,resized)
    cv2.imshow("lakes",photo)
    cv2.waitKey()
    cv2.destroyAllWindows()
    dest=upload + "/image{}.png".format(count)
    

    black=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dest,black)