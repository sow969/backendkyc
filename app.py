import cv2
import re
import pytesseract
import numpy as np
from PIL import Image
import os
from flask import *
import json
from flask_cors import CORS,  cross_origin
import mysql.connector as conn
from flask_mysqldb import MySQL
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'bd_kyc'
UPLOAD_FOLDER = 'Images/documentUpload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
mysql = MySQL(app)
CORS(app)

db = []
known_path = os.path.join(os.getcwd(),"Images/visageConnue/")
unknown_path = os.path.join(os.getcwd(), "Images/VisageInconnue/")

path ='Images/visageConnue'

pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def get_data():
    global db
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM td_utilisateur")   
    result = cur.fetchall()
    for i in result:     
        l = []        
        l.append(i[10])    
    cur.close()    
    cur.close()



@app.route("/utilisateur/enroleUser", methods=["POST", "OPTIONS"])
def enroleUser():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()
    elif request.method == 'POST':
        # Prenom = request.form.get("prenom")
        # Nom = request.form.get("nom")
        # Email = request.form.get("email")
        # Telephone = request.form.get("telephone")
        # Cni = request.form.get("cni")
        fs = request.files.get('photo')
        count = 0
        if fs:
            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 8)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                cv2.imwrite("Images/visageConnue/"+""+ '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                if count>3:
                    break
            # cur = mysql.connection.cursor()
            # cur.execute('INSERT INTO td_utilisateur(prenom, nom, telephone, email, cni)VALUES(%s, %s, %s, %s, %s)',(Prenom,Nom,Telephone,Email,Cni))
            # mysql.connection.commit()
            # cur.close()
            return jsonify(data="enrollement reusi!!")
        else:
            return jsonify(data="Viellez ajouter votre photo!")

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/utilisateur/loginVisage", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()
    elif request.method == "POST":
        fs = request.files.get('photo')
        traitement(path)
        if fs:
            font = cv2.FONT_HERSHEY_PLAIN
            recognizer.read('Images/trainner/trainer.yml')
            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2,8)
            counter = 0
            for(x,y,w,h) in faces:
                Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if confidence <40:
                     print(confidence)
                     succes=True
                     #cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
                     #cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
                else:
                    Id = "personne inconnue"
                    succes=False
                    #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,0,255), 4)
                    #cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)
                    #cv2.putText(img, str(Id), (x,y-40), font, 2, (0,0,255), 3)
            #cv2.imshow('frame',im) 

        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        # cam.release()
        # cv2.destroyAllWindows() 
        return jsonify(data = succes)
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def traitement(path):
    print(path)
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
    return faceSamples,ids

faces,ids = traitement('Images/visageConnue')
recognizer.train(faces, np.array(ids))
recognizer.save('Images/trainner/trainer.yml')


# Service pour la recuperation des informations d'un CNI
@app.route("/inscription/imageUpload", methods=["POST", "OPTIONS"])
def uploadImage():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()
    elif request.method == 'POST':
        fs = request.files.get('document')
        count = 0
        if fs:
            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            im2 = img.copy()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = im2[y:y + h, x:x + w]
                text = pytesseract.image_to_string(cropped)
                cni = text[5:14]
                cni1 = text[15:23]
                cni2 = text
            faces = detector.detectMultiScale(gray, 1.2, 8)
            donnee=cni2.split('\n')
            i = donnee.index('Prenoms')
            j = donnee.index('Nom')
            k = donnee.index('Date de naissar Sexe')
            l = donnee.index('lieu de naiss:')
            m = donnee.index("N° de la carte d'identite")
            n= donnee.index('Adresse du domicile')
            o = donnee.index("Date sie \délivrar Date d'expiration.")
            p = donnee.index('Oeive denregistrement')
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                count += 1
                cv2.imwrite("Images/visageConnue/"+cni1+ '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                if count>3:
                    break
            return jsonify(prenom=donnee[i+2],Nom=donnee[j+2],
             date_naissance=donnee[k+1], Lieu_naissance=donnee[l+2],
             numero_cni=donnee[m+2], Adresse_domicile=donnee[n+2],
             date_Expiration=donnee[o+2], centre_enregistrement=donnee[p+1])
        else:
            return 'Inserer le document svp!'


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# endpoint de traitement empreinte digitale
@app.route("/utilisateur/empreinte", methods=["POST", "OPTIONS"])
def empreinteUser():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()
    elif request.method == 'POST':
        fs = request.files.get('empreinte')
        count = 0
        if fs:
            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            best_score = 0
            filename = None
            image = None
            kp1, kp2, mp = None, None, None
            for file in [file for file in os.listdir("empreinteDigitale")][:1000]:
                fingerprint_image = cv2.imread("empreinteDigitale/"+file)
                sift = cv2.SIFT_create()
                keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)
                keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image,None)
                matches = cv2.FlannBasedMatcher({'algorithm':1, 'trees':10},{}).knnMatch(descriptors_1, descriptors_2, k = 2)

                match_points = []
                for p, q in matches:
                    if p.distance < 0.1*q.distance:
                        match_points.append(p)
                keypoints = 0
                if len(keypoints_1) < len(keypoints_2):
                    keypoints = len(keypoints_1)
                else:
                    keypoints = len(keypoints_2)
                if len(match_points)/ keypoints*100>best_score:
                    best_score = len(match_points)/keypoints*100
                    filename = file
                    image = fingerprint_image
                    kp1, kp2, mp = keypoints_1, keypoints_2, match_points
                    succes = True
                    return jsonify(data=succes)

                else:
                    succes = False
                    return jsonify(data =succes)
        else:
            return jsonify(data="Viellez ajouter votre empreinte svp!")

@app.route("/")
def index():
    return "L'api a demarré!!"
app.run(threaded=True, port=5000)
