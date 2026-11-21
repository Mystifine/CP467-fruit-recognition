import cv2
import numpy as np 
IMG_SIZE = 100
SAT_THRESH = 40      
VAL_THRESH = 220 

KERNEL = np.ones((5, 5), np.uint8)



def load_image(path): #read image then resize it 

    
    if path is not None: 
        img = cv2.imread(str(path))
        if img is None:
            raise NotImplementedError(f"Image not found at path: {path}")
        
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
   




def segment(img_bgr):
   

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) #HSV conversion and thresholding 
    mask = np.where((hsv[:, :, 1] > SAT_THRESH) | (hsv[:, :, 2] < VAL_THRESH), 255, 0).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL) #using morphology to remove noise and fill holes of the images

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((IMG_SIZE,IMG_SIZE), dtype = np.uint8), None  #finding outlier contours

    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8) #keep largest contour and fill it on a blank mask
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)

    return filled_mask, largest_contour 



def draw_result(img_bgr, contour, label, correct=None, scale=3):

   #resize image by scale using nearest neighbor interpolation to avoid blurring the image
    result = cv2.resize(img_bgr,(0, 0),fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) 
    scaled = (contour * scale).astype(np.int32)

    cv2.drawContours(result, [scaled], -1, (0, 255, 0), 2)
    #generate a rotated bounding box around the contour and draw it on the image. 
    box = cv2.boxPoints(cv2.minAreaRect(scaled)).astype(np.int32)
    cv2.polylines(result, [box], True, (255, 0, 255), 2)


   #set colour text based on correct parameter 
    if correct is True:
        color = (0, 255, 0)  #green 
    elif correct is False:
        color = (0, 0, 255) #red
    else:
        color = (255, 0, 0) #blue
 
    x = int(np.clip(box[:, 0].min(), 0, result.shape[1] - 1))
    y = int(max(box[:, 1].min() - 8, 20))
    cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result

if __name__ == "__main__":

    img = np.full((IMG_SIZE, IMG_SIZE, 3), 250, np.uint8)
    cv2.circle(img, (50, 50), 35, (30, 140, 250), -1)
    cv2.circle(img, (40, 40), 5, (255, 255, 255), -1)
    mask, contour = segment(img)
    area = int((mask > 0).sum())
    print(f"area {area} (expect ~3850), highlight filled: {mask[40, 40] == 255}")
    assert contour is not None and 3500 < area < 4200 and mask[40, 40] == 255
    print("ok")