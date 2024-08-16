
import base64
import cv2
import numpy as np
from flask import Flask, render_template, session, request, jsonify, send_file
from flask_cors import CORS
import os
# import pytesseract
import pandas as pd
import csv
import matplotlib.pyplot as plt
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image

# import keras_ocr

ocr_model = ocr_predictor(pretrained=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
csv_path = 'output.csv'

# pipeline = keras_ocr.pipeline.Pipeline()
    
app = Flask(__name__)
# CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    try:
        # Get the image data from the request
        image_data = request.form.get('image')
        
        if image_data:
            # Decode the base64 string into bytes
            image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
            image_bytes = base64.b64decode(image_data)

            # Convert bytes to a NumPy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode the image array into an OpenCV image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            pil_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "camera.png"))
            try:
                text = detect_character("camera.png")
                print(text)            
                SNdata = searchData("SN:", text)
                if SNdata=="":
                    SNdata = searchData("S/N", text)
                username = searchData("User name", text)
                if username == "":
                    username = searchData("Username", text)
                
                wifiPassword = searchData("WiFi password", text)
                if wifiPassword == "":
                    wifiPassword = searchData("WifiPassword", text)
                if wifiPassword == "":
                    wifiPassword = searchData("wi-fi password", text)

                password = searchData("Password", text)      
                
                if password == wifiPassword:
                    password = searchData("Password", text, retry_search = True) 
                    
                httpAddress = searchData("http://", text)
                if httpAddress == "":
                    httpAddress = searchData("ttp://", text)
                if httpAddress == "":
                    httpAddress = searchData("ttp:/", text)
                if httpAddress == "":
                    httpAddress = searchData("ttp/", text)
                
                df_new = pd.DataFrame({
                    'SNdata': [SNdata],
                    'username': [username],
                    'password': [password],
                    'wifiPassword': [wifiPassword],
                    'httpAddress': [httpAddress]
                })

                csv_path = 'output.csv'
                
                # Check if the file already exists
                if not os.path.isfile(csv_path):
                    # If it doesn't exist, create a new one
                    df_new.to_csv(csv_path, index=False)
                else:
                    # If it exists, load the existing data into a DataFrame
                    df_existing = pd.read_csv(csv_path)
                    # Append the new data to the existing data
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    # Save the combined data back to the CSV file
                    df_combined.to_csv(csv_path, index=False)

                # with open(csv_path, 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerows([[SNdata], [username], [password], [wifiPassword], [httpAddress]])
                return jsonify({'message': f'S/N: {SNdata}, UserName: {username}, Password: {password}, Wifi Password : {wifiPassword}, Http Address : {httpAddress}'})
            except:
                return jsonify({'message': 'no character detected'})
        else:
            return jsonify({'message': 'No image data provided'})

    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/download')
def download_csv():
    if not os.path.isfile(csv_path):
        return jsonify({'message': 'No CSV file found'}), 404
    return send_file(csv_path, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = detect_character(filename)
        SNdata = searchData("SN:", text)
        if SNdata=="":
            SNdata = searchData("S/N", text)
        username = searchData("User name", text)
        if username == "":
            username = searchData("Username", text)
        
        wifiPassword = searchData("WiFi password", text)
        if wifiPassword == "":
            wifiPassword = searchData("WifiPassword", text)
        if wifiPassword == "":
            wifiPassword = searchData("wi-fi password", text)

        password = searchData("Password", text)      
        
        if password == wifiPassword:
            password = searchData("Password", text, retry_search = True) 

        httpAddress = searchData("http://", text)
        if httpAddress == "":
            httpAddress = searchData("ttp://", text)
        if httpAddress == "":
            httpAddress = searchData("ttp:/", text)
        if httpAddress == "":
            httpAddress = searchData("ttp/", text)
        
        
        df_new = pd.DataFrame({
                'SNdata': [SNdata],
                'username': [username],
                'password': [password],
                'wifiPassword': [wifiPassword],
                'httpAddress': [httpAddress]
            })

        csv_path = 'output.csv'
        
        # Check if the file already exists
        if not os.path.isfile(csv_path):
            # If it doesn't exist, create a new one
            df_new.to_csv(csv_path, index=False)
        else:
            # If it exists, load the existing data into a DataFrame
            df_existing = pd.read_csv(csv_path)
            # Append the new data to the existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Save the combined data back to the CSV file
            df_combined.to_csv(csv_path, index=False)
        
        return jsonify({'message': f'S/N: {SNdata}, UserName: {username}, Password: {password}, Wifi Password : {wifiPassword}, Http Address : {httpAddress}'})
    else:
        return jsonify({'error': 'File type not allowed'}), 400

def compare_strings(str1, str2):
    # Check if they are totally same
    if str1 == str2:
        return 0
    
    len1, len2 = len(str1), len(str2)

    # If lengths differ by more than 1, return 2 immediately
    if abs(len1 - len2) > 1:
        return 2

    # Counter for differences
    diff_count = 0

    i, j = 0, 0
    while i < len1 and j < len2:
        if str1[i] != str2[j]:
            if diff_count == 1:  # If already found one difference
                return 2

            # Increase difference counter
            diff_count += 1

            # Check if it's a character addition or omission
            if len1 > len2:
                i += 1
            elif len1 < len2:
                j += 1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1
    
    # Handle the end of the loop case where one character is added/omitted at the end
    if i < len1 or j < len2:
        diff_count += 1

    return 1 if diff_count == 1 else 2

# Example usage


# def searchData(substr, source, exceptionstr="", nextsearch=""):
    
#     for line in source:
#         if exceptionstr != "":
#             if exceptionstr.upper() in line.upper():
#                 continue
#         if substr.upper() in line.upper():
#             index = line.upper().find(substr)
#             if len(line) - index - len(substr) > 3:
#                 tempstr = line[index + len(substr) + 1:].split(" ")
#                 for temp in tempstr:
#                     if len(temp) > 3:
#                         return temp
#             flag = True
#     return ""

def searchData(substr, source, retry_search = False):
    if substr.upper() in source.upper():
        index = source.upper().find(substr.upper())
        tempstr = source[index+len(substr) : ]
        if retry_search:
            if substr.upper() in tempstr.upper():
                index = tempstr.upper().find(substr.upper())
                newstr = tempstr[index+len(substr) : ]
                tempstr = newstr
            else:
                return ""
        for temp in tempstr.split(" "):
            if len(temp)>=3:
                return temp
    return ""

def detect_character(source_file):
    if isinstance(source_file, str):
        img = os.path.join(app.config['UPLOAD_FOLDER'], source_file)
        
    else:
        img = source_file
    
    doc = DocumentFile.from_images(img)

    result = ocr_model(doc)
    # print(result)
    lines = []
    text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_contents = ""
                for word in line.words:
                    if len(word.value)<=3:
                        continue
                    line_contents += word.value + " "
                text += line_contents
                lines.append(line_contents)
    return text


# def keras_detect_character(filename):
#     # keras-ocr will automatically download pretrained
#     # weights for the detector and recognizer.
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     # img = cv2.imread(filepath)
#     # Step 2: Preprocess the image
#     # rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     image = keras_ocr.tools.read(filepath)

#     predictions = pipeline.recognize([image])

#     # Extract the detected boxes and texts
#     boxes, texts = predictions[0]

#     return texts

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(host='0.0.0.0',  ssl_context=('cert.pem', 'key.pem'))

