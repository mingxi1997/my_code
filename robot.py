from pyzbar import pyzbar
import face_recognition
import cv2
import numpy as np
import time
import datetime
import subprocess 
# from multiprocessing import Queue, Process
import threading
import socket
HOST = "192.168.149.1"
PORT = 1065

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
        
with open("./lib/coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./lib/yolov3.cfg"
modelWeights = "./lib/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    
    obj=[]
    
    
    obj.append(label)
    obj.append((left, top - round(1.5*labelSize[1]), left + round(1.5*labelSize[0]), top + baseLine))
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    return obj

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    objs=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        obj=drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        objs.append(obj)
    return objs
  
 
        
def Qc_code(frame):
    # global width, height
    barcodeData = None
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray_frame)
    
    for barcode in barcodes:
        x, y, w, h = barcode.rect
      
        barcodeData = barcode.data.decode("utf-8")
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
 
        print(barcodeData)

    return frame,barcodeData

def play_mp3(path): 
    subprocess.Popen(['mpg123', '-q', path]).wait()
    
index=0
def sound_play():
    global index
    global find
    while not find:
        if index==1:
            play_mp3('ren.mp3')
            index=0
            time.sleep(6)
        else:
            pass
        time.sleep(1)
    
        
url = "http://192.168.149.1:8080/?action=stream?dummy=param.mjpg"
# url=0

cap = cv2.VideoCapture(url)


my_image = face_recognition.load_image_file("1.jpg")
my_encoding = face_recognition.face_encodings(my_image)[0]
known_face_encodings = [my_encoding]
known_face_names = ["ME"]
show=0

cmd = "I003-" + str(9) + "-1\r\n"  # 右转 小步伐


def finding():
    global interval
    global find
    global status
    while True:
        if not find and (status=='face' or status=='obj'):
            
            time.sleep(interval)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # 创建客户端
            client.connect((HOST, PORT))
            client.send(cmd.encode())
            client.close()
        else:
            pass
       

find=False
interval=4

status='zbar'

thread_finding = threading.Thread(target=finding)
thread_finding.start()   
while True:
  if status=='zbar':
         find=False
         print('start zbar')
         ret,frame=cap.read()
         frame,barcodeData=Qc_code(frame)
         print(barcodeData)
         cv2.imshow('Video', frame)
         cv2.waitKey(1)
         if barcodeData=='face':
             status='face'
             cv2.destroyAllWindows()
         elif barcodeData=='obj':
             status='obj'
             cv2.destroyAllWindows()
         
         
         
  while not find and status=='face':

    
    ret,frame=cap.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame,model="cnn")
    print(datetime.datetime.now())
    print(face_locations)
    if face_locations==[]:
        interval=4
    else:
        interval=8   
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)    
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        print(face_distances)
        print(top, right, bottom, left)
        if face_distances[0]<0.4:
            name='xu'
            find=True
            # thread_finding.join()
            status='zbar'
            print('i find you')
            cv2.destroyAllWindows()
            break
     

        else:
            print('not you')
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)        
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
  while not find and status=='obj':
    
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video


    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    objs=postprocess(frame, outs)
    
    names=[obj[0].split(':')[0] for obj in objs]
    if 'bottle' in names:
            find=True
            # thread_finding.join()
            status='zbar'
            print('i find it')
            cv2.destroyAllWindows()
            break
        

    cv2.imshow('frame', frame)
    cv2.waitKey(1)



