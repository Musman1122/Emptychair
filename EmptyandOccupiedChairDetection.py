import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4-custom_4000.weights" , "yolov4-custom.cfg")
classes = []
with open("D:/ESS/Emptyandoccupiedchairdetection/predefined_classes.txt" , "r") as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes)    
# print(len(classes))
layer_names = net.getLayerNames()
output_layer = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("Image_801.jpg")
print(img.shape)
img = cv2.resize(img , None , fx = 3.55 , fy = 2.55)  #25% by default hota hai then jab resize krty hai tu 25% or barha daita hai
height , width ,channels = img.shape
print(img.shape)


#for object display
cv2.imshow("ESS" , img)
cv2.waitKey(0) #stable rakhta hai
cv2.destroyAllWindows()

#1 -------->grey scale
# 1 2 3------------->red blue or green

#detection objects
blob = cv2.dnn.blobFromImage(img , 0.00392 , (416 , 416) , (0 , 0 , 0) , True , crop = False)

#0.00392 total pixels _______0 - 255 (1/255 = 0.00392)
#prigin = (0,0,0)

net.setInput(blob)  #input du ga net layer ko
outs =  net.forward(output_layer)

#showing information on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores) #score ks object ka hai aur class name bta dein
        confidence = scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            #cv2.circle(img , (center_x , center_y) , 25 , (255 , 0,0) , 5)   # blue green red  # 25 radius    
            
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            #rectangle coordinates
            
            x = int(center_x - w /2)  #starting mei aa jai ge pehlay 2 sy divide karay then - center than start mei aa jai ga
            y = int(center_y - h /2)
            
            boxes.append([x , y , w , h])
            confidences.append(float(confidence))
            class_ids.append(class_id)            
            
indexes = cv2.dnn.NMSBoxes(boxes , confidences , 0.3 , 0.2) 
print(len(indexes))

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x , y , w , h = boxes[i]
        label = str(classes[class_ids[i]])
        Accuracy = str(int(round(confidences[i]*100 , 0))) + "%"
        cv2.rectangle(img , ( x , y) , (x + w , y+h) , (0 , 255 , 0) , 4)
        cv2.putText(img ,label + ":" + Accuracy , (x , y + 25) , font , 1 , (255 , 0 , 0) , 2)
          
            
#for show the circles in image            
cv2.imshow("ESS" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()



         
  
            














    