import cv2
import numpy as np
import tensorflow as tf
import time
from pynput import keyboard
from pynput.keyboard import Key
import socket

#Face haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Define max values for left/right pupils
top_max_left = 5000
left_max_left = 5000
right_max_left = 0
bottom_max_left = 0
top_center_max_left = 5000
left_center_max_left = 5000
right_center_max_left = 0
bottom_center_max_left = 0
top_max_right = 5000
left_max_right = 5000
right_max_right = 0
bottom_max_right = 0
top_center_max_right = 5000
left_center_max_right = 5000
right_center_max_right = 0
bottom_center_max_right = 0
centerpointsList = []
leftgazeCenterX = 0
box_array = []
max_count_bool = True

def centerpointFunction(left, right, top, bottom):
  pupilBoxWidth = right - left
  pupilBoxHeight = bottom - top
  pupilCenterX = pupilBoxWidth/2 + left
  pupilCenterY = pupilBoxHeight/2 + top
  centerPointsList = [pupilCenterX,pupilCenterY]
  return centerPointsList

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def calculate_overlap(box_prew, box_actual):
    top_prew = box_prew[0]
    left_prew = box_prew[1]
    bottom_prew = box_prew[2]
    right_prew = box_prew[3]
    top_act = box_actual[0]
    left_act = box_actual[1]
    bottom_act = box_actual[2]
    right_act = box_actual[3]

    w_prew = right_prew - left_prew
    h_prew = bottom_prew - top_prew
    w_act = right_act - left_act
    h_act = bottom_act - top_act

     # Calculate the coordinates of the intersection rectangle
    x_left = max(left_prew, left_act)
    y_top = max(top_prew, top_act)
    x_right = min(right_prew, right_act)
    y_bottom = min(bottom_prew, bottom_act)

    # Check for intersection
    if x_right < x_left or y_bottom < y_top:
        return 0

    # Calculate the area of each box
    box1_area = w_prew * w_act
    box2_area = h_prew * h_act

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the percentage of overlap based on the minimum area
    overlap_percent = min(intersection_area / box1_area, intersection_area / box2_area) * 100

    return overlap_percent


# flag to indicate whether a client is connected
client_connected = False
# flags to detect which pupil is moving
left_pupil_is_moveing = False
right_pupil_is_moveing = False

# ESP32 URL
URL = "http://10.61.3.72"
AWB = True

# Load TFLite model
model_path = "android (1).tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the video file
#cap = cv2.VideoCapture(URL + ":81/stream")
#cap = cv2.VideoCapture("eyetest.mp4")
#cap = cv2.VideoCapture("eyetest2.mkv")
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("asd.mkv")

# create a TCP/IP socket and bind it to a port
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('', 8888)
#server_address = ('localhost', 8888)
sock.bind(server_address)
print("Server adress: " ,server_address)

# listen for incoming connections
sock.listen(1)
print("Server started on port", server_address[1])

print(sock)

# wait for a client to connect
print("Waiting for a client to connect at start...")
client_sock, client_address = sock.accept()
print("Client connected start:", client_address)
print("Client sock:", client_sock)
client_connected = True

while True:

    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break
    
    start_time = time.time()

    frame = cv2.flip(frame, 1)
    """
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        hhalf = int(h/2)
        frame = frame[y+65:y+100, x+45:x+165]

    h, w, c = frame.shape
    dim = (w*3, h*3)
    try:
        frame = zoom(frame,8)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
    except:
        pass
    """
    frame = frame[220:280,230:360]
    #frame = frame[220:280,220:280]
    frame = zoom(frame,6)

    #Image enhancement strategies:
    
    # Adjust the brightness and contrast
    # Adjusts the brightness by adding 10 to each pixel value
    brightness = 4 
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 1.1  
    frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)
    # Create the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Sharpen the image
    frame = cv2.filter2D(frame, -1, kernel)
    #frame = cv2.GaussianBlur(frame, (9, 9), 0)
    #

    #dim = (400, 200)
    #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


    # Pre-process input image
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.array(input_data, dtype=np.uint8)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the model
    interpreter.invoke()

    # Get the output
    detection_scores = interpreter.get_tensor(output_details[0]['index'])
    detection_boxes = interpreter.get_tensor(output_details[1]['index'])
    detection_time = interpreter.get_tensor(output_details[2]['index'])
    detection_class = interpreter.get_tensor(output_details[3]['index'])
    
    #cv2.imshow("output", frame)
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    num_detections = 0
    for i in range(detection_boxes.shape[1]):
        if num_detections >= 2:
            break

        if detection_scores[0, i] > 0.5:
            num_detections += 1
            box = detection_boxes[0, i, :] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            (top, left, bottom, right) = box.astype("int")
            if(right < frame.shape[1]/2):
                if(right-left>30 and bottom-top>30):
                    
                    #Calculating the overlap between the previous and the actual bounding boxes
                    bouding_box = [top,left,bottom,right]
                    box_array.append(bouding_box)

                    overlap_percent = None
                    if(len(box_array)==2):
                        overlap_percent = calculate_overlap(box_array[0],box_array[1])
                        print(overlap_percent)

                        if(overlap_percent != None and overlap_percent>90): #if overlap bigger than given percent do not redraw
                            cv2.rectangle(frame, (box_array[0][1], box_array[0][0]), (box_array[0][3], box_array[0][2]), (0, 255, 0), 2)
                            pass
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        box_array.pop(0)

                   
                    if(top_max_left>top):
                        top_max_left = top
                    if(left_max_left>left):
                        left_max_left = left
                    if(bottom_max_left<bottom):
                        bottom_max_left = bottom
                    if(right_max_left < right):
                        right_max_left = right

                    #cv2.rectangle(frame, (left_max-20, top_max-10), (right_max+20, bottom_max+10), (255, 0, 0), 2)
                    
                    #Box max width and height
                    #cv2.rectangle(frame, (left_max_left, top_max_left), (right_max_left, bottom_max_left), (255, 0, 0), 2)
                    
                    left_pupil_is_moveing = True
            elif(right > frame.shape[1]/2):
                if(right-left>30 and bottom-top>30):

                #Calculating the overlap between the previous and the actual bounding boxes
                    bouding_box = [top,left,bottom,right]
                    box_array.append(bouding_box)

                    overlap_percent = None
                    if(len(box_array)==2):
                        overlap_percent = calculate_overlap(box_array[0],box_array[1])
                        print(overlap_percent)

                        if(overlap_percent != None and overlap_percent>90): #if overlap bigger than given percent do not redraw
                            cv2.rectangle(frame, (box_array[0][1], box_array[0][0]), (box_array[0][3], box_array[0][2]), (0, 255, 0), 2)
                            pass
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        box_array.pop(0)

                    if(top_max_right>top):
                        top_max_right = top
                    if(left_max_right>left):
                        left_max_right = left
                    if(bottom_max_right<bottom):
                        bottom_max_right = bottom
                    if(right_max_right < right):
                        right_max_right = right

                    #cv2.rectangle(frame, (left_max-20, top_max-10), (right_max+20, bottom_max+10), (255, 0, 0), 2)
                    
                    #Box max width and height
                    #cv2.rectangle(frame, (left_max_right, top_max_right), (right_max_right, bottom_max_right), (255, 0, 0), 2)
                    
                    right_pupil_is_moveing = True
                    left_pupil_is_moveing = False
            else:
                left_pupil_is_moveing = False
                right_pupil_is_moveing = False
            #print("Detection time: ", int(detection_time))
            #print("Detection class: Pupil")


    if(left_pupil_is_moveing):
        #Calculate center points of pupil
        centerpointsList = centerpointFunction(right, left, top, bottom)
        frame = cv2.circle(frame, ((int(centerpointsList[0])),(int(centerpointsList[1]))), radius=5, color=(255, 255, 255), thickness=1)
        
        width = right-left
        frame_width = frame_width/2
        max_width = int(frame_width*0.7)
        height = bottom-right
        frame_height = frame_height/2
        max_height = int(frame_height*0.7)

        if(width<max_width and height<max_height and max_count_bool):
            if(top_center_max_left>int(centerpointsList[1])):
                top_center_max_left = int(centerpointsList[1])
            if(left_center_max_left>int(centerpointsList[0])):
                left_center_max_left = int(centerpointsList[0])
            if(bottom_center_max_left<int(centerpointsList[1])):
                bottom_center_max_left = int(centerpointsList[1])
            if(right_center_max_left<int(centerpointsList[0])):
                right_center_max_left = int(centerpointsList[0])
        cv2.rectangle(frame, (left_center_max_left, top_center_max_left), (right_center_max_left, bottom_center_max_left), (255, 255, 0), 2)
        #print(len(centerpointsList))
        #print(centerpointsList[0])

        #leftPupilMaxWidth = right_max+20-left_max-20
        #leftPupilMaxHeight = bottom_max+10-top_max-10
        leftPupilMaxWidth = right_center_max_left-left_center_max_left
        leftPupilMaxHeight = bottom_center_max_left-top_center_max_left
        #leftPupilWidth = right-left
        #leftPupilHeight = bottom-top
        #leftgazeWidth = 1024
        #leftgazeHeight = 600
        leftgazeWidth = 1980
        leftgazeHeight = 1080
        leftpupilCenterX = centerpointsList[0]-left_center_max_left
        leftpupilCenterY = centerpointsList[1]-top_center_max_left
        try:
            leftwPercent = (leftpupilCenterX/leftPupilMaxWidth)*100
        except:
            print("rightwPercent wrong value")
        #hPercent = 100-(pupilCenterY/leftPupilMaxHeight)*100
        try:
            lefthPercent = (leftpupilCenterY/leftPupilMaxHeight)*100
        except:
            print("righthPercent wrong value")
        leftgazeCenterX = (leftwPercent/100)*leftgazeWidth
        leftgazeCenterY = (lefthPercent/100)*leftgazeHeight

        #print("hpercent: ",hPercent)
        #print("gazeCenterY: ",gazeCenterY)
        #print("pupilCenterY: ",pupilCenterY)
    elif(right_pupil_is_moveing):
        #Calculate center points of pupil
        centerpointsList = centerpointFunction(right, left, top, bottom)
        frame = cv2.circle(frame, ((int(centerpointsList[0])),(int(centerpointsList[1]))), radius=5, color=(255, 255, 255), thickness=1)
        
        width = right-left
        frame_width = frame_width/2
        max_width = int(frame_width*0.7)
        height = bottom-right
        frame_height = frame_height/2
        max_height = int(frame_height*0.7)

        if(width<max_width and height<max_height and max_count_bool):
            if(top_center_max_right>int(centerpointsList[1])):
                top_center_max_right = int(centerpointsList[1])
            if(left_center_max_right>int(centerpointsList[0])):
                left_center_max_right = int(centerpointsList[0])
            if(bottom_center_max_right<int(centerpointsList[1])):
                bottom_center_max_right = int(centerpointsList[1])
            if(right_center_max_right<int(centerpointsList[0])):
                right_center_max_right = int(centerpointsList[0])
        cv2.rectangle(frame, (left_center_max_right, top_center_max_right), (right_center_max_right, bottom_center_max_right), (255, 255, 0), 2)
        #print(len(centerpointsList))
        #print(centerpointsList[0])

        #leftPupilMaxWidth = right_max+20-left_max-20
        #leftPupilMaxHeight = bottom_max+10-top_max-10
        rightPupilMaxWidth = right_center_max_right-left_center_max_right
        rightPupilMaxHeight = bottom_center_max_right-top_center_max_right
        #leftPupilWidth = right-left
        #leftPupilHeight = bottom-top
        rightgazeWidth = 1980
        rightgazeHeight = 1080
        #rightgazeWidth = 1024
        #rightgazeHeight = 600
        rightpupilCenterX = centerpointsList[0]-left_center_max_right
        rightpupilCenterY = centerpointsList[1]-top_center_max_right
        try:
            rightwPercent = (rightpupilCenterX/rightPupilMaxWidth)*100
        except:
            print("rightwPercent wrong value")
        #hPercent = 100-(pupilCenterY/leftPupilMaxHeight)*100
        try:
            righthPercent = (rightpupilCenterY/rightPupilMaxHeight)*100
        except:
            print("righthPercent wrong value")
        rightgazeCenterX = (rightwPercent/100)*rightgazeWidth
        rightgazeCenterY = (righthPercent/100)*rightgazeHeight

        #print("hpercent: ",hPercent)
        #print("gazeCenterY: ",gazeCenterY)
        #print("pupilCenterY: ",pupilCenterY)

    # Show the output
    dim = (300, 100)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("output", frame)

    try:
        if not client_connected:
            # wait for a client to connect
            print("Waiting for a client to connect...")
            client_sock, client_address = sock.accept()
            print("Client connected:", client_address)

            # set the flag to indicate that a client is connected
            client_connected = True
    except:
        print("Client not connected")
        
    # check if the client is still connected
    try:
        #if(left_pupil_is_moveing and right_pupil_is_moveing==False):
        if(left_pupil_is_moveing):
            # send response back to the client
            #message = str(ymin) + ","+ str(ymax) + ","+ str(xmin) + ","+ str(xmax)
            message = str(leftgazeCenterX) + "," + str(leftgazeCenterY)
            client_sock.sendall(message.encode())
        elif(right_pupil_is_moveing and left_pupil_is_moveing==False):
            # send response back to the client
            #message = str(ymin) + ","+ str(ymax) + ","+ str(xmin) + ","+ str(xmax)
            message = str(rightgazeCenterX) + "," + str(rightgazeCenterY)
            client_sock.sendall(message.encode())
        elif(right_pupil_is_moveing and left_pupil_is_moveing):
            # send response back to the client
            #message = str(ymin) + ","+ str(ymax) + ","+ str(xmin) + ","+ str(xmax)
            gazeCenterXmean = (leftgazeCenterX+rightgazeCenterX)/2
            gazeCenterYmean = (leftgazeCenterY+rightgazeCenterY)/2
            message = str(gazeCenterXmean) + "," + str(gazeCenterYmean)
            client_sock.sendall(message.encode())
        else:
            if(leftgazeCenterX == 0):
                message = str(500) + "," + str(500)
            else:
                message = str(leftgazeCenterX) + "," + str(leftgazeCenterY)
            client_sock.sendall(message.encode())
        
        # receive data from the client, send response back, and close the connection
        data = client_sock.recv(1024)
        #print("Received:", data.decode())
        if(data.decode() == "reset_max_values"):
            top_max_left = 5000
            left_max_left = 5000
            right_max_left = 0
            bottom_max_left = 0
            top_center_max_left = 5000
            left_center_max_left = 5000
            right_center_max_left = 0
            bottom_center_max_left = 0
            top_max_right = 5000
            left_max_right = 5000
            right_max_right = 0
            bottom_max_right = 0
            top_center_max_right = 5000
            left_center_max_right = 5000
            right_center_max_right = 0
            bottom_center_max_right = 0
            #print("Runned")
            #cv2.waitKey(0)
        if(data.decode() == "continue"):
            max_count_bool = True
            pass
        if(data.decode() == "stop_max_count"):
            max_count_bool = False
            pass
            #print("continued")
        #message = "Thanks for the message!"
    except:
        # client is disconnected
        print("Client disconnected")
        client_connected = False

    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    #print("Total time in milliseconds:", str(int(total_time))+ " ms")

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video and destroy all windows
cap.release()
# close the connection
client_sock.close()
cv2.destroyAllWindows()