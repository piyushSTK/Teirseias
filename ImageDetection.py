from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
f=open('text.txt','w')
for eachObject in detections:
	#print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
	f.write(eachObject["name"]+"\n")
f.close()
c=open('text.txt','r')
text2=c.read()
c.close
print(text2)

