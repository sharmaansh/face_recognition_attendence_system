from pygame import mixer
from tkinter import *
from tkinter import filedialog
import cv2
import os
import numpy as np
from PIL import Image
import smtplib
import pandas as pd
import time
import sqlite3

window=Tk()#likegeeks.com--tkinter
window.title('FDM')

def dataset():

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_id = input('enter user id----->  ')

    print("Initializing:Look the camera and wait ...")

    count = 0

    while(True):

        ret, img = cam.read()
        img = cv2.flip(img, 1) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(80) & 0xff 
        if k == 27:
            break
        elif count >=100:
             break
    print("Exiting.........")
    cam.release()
    cv2.destroyAllWindows()
def train():

    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
                cv2.imshow("Adding faces",img_numpy[y:y+h,x:x+w])
                cv2.waitKey(4)

        return faceSamples,ids

    print ("Training faces..............")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    cv2.destroyAllWindows()
    print("{0} faces trained.".format(len(np.unique(ids))))

def detect():

    from datetime import date
    def sendmail(mailid,sms):
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login("mail@gmail.com", "password.")
        text=(sms)
        print('message sent is:--> '+text)
        server.sendmail('mail@gmail.com',mailid,text)
    df=pd.read_csv('atten.csv')
    d=df['name']

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0

    li=np.array(df['mail'])
    names = df['name']

    print('starting camera.......')
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    print ('detecting face.....')
    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if confidence<55:
                
                idno=id
                li=np.insert(li,id,names[id],axis=0)
                id = names[id]
    ##          confidence = "  {0}%".format(round(100 - confidence))
            else:
                idno=0
                
            cv2.putText(img, 'Name: '+str(df['name'][idno]), (x+5,y-55), font, 1, (255,255,255), 2)
            cv2.putText(img, 'Sub: '+str(df['subj'][idno]), (x+5,y-30), font, 1, (255,255,255), 2)
            cv2.putText(img, 'age: '+str(df['age'][idno]), (x+5,y-5), font, 1, (255,255,255), 2)
    ##        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        cv2.imshow('camera',img)
        
        #print li
        
        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break
        
    print("Exiting camera Program")
    cam.release()
    cv2.destroyAllWindows()
    da=str(date.today())

    l1=np.unique(li)
    print ('detected face id:-->'+l1)
    print('\n')
    print(df)
    print('\n')
    for n in l1:
        for f in range(0,len(df['name'])):
            if n==df['name'][f]:
                #df1.set_value(f,da,'present')
                df.ix[f,da]='present'
    print('\n')
    print(df)
    print('\n')
    df.to_csv('atten.csv',index=False)

    for data in l1:
        testdata=data
        i=0
        for n in d:
            if n==testdata:
                print('sending mail to:',testdata)
                m=df['mail'][i]
                print('\n')
                sendmail(m,('subject: facial recognition attendence \r\n'+'mr '+df['name'][i]+' is present on date '+da+' thank you'))
                print('\n')
                print('mail sent to:',testdata)
            else:
                pass
            i +=1
    conn = sqlite3.connect('tutorial.db')
    c = conn.cursor()
    df = pd.read_csv('atten.csv')
    df.to_sql(0, conn, if_exists='replace', index=False)
    conn.commit()
    c.close
    conn.close()
    print('csv file created')
    print('database created')
                
btn1=Button(window,text='Data Set',command=dataset)
btn2=Button(window,text='Train',command=train)
btn3=Button(window,text='Detection',command=detect)

btn1.grid(column=1,row=0)
btn2.grid(column=2,row=0)
btn3.grid(column=3,row=0)
    
window.mainloop()



