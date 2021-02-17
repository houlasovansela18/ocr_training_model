import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import queue
import threading
import time
import os

from core.local_utils import detect_lp
from absl import app, flags, logging
from os.path import splitext,basename
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher



def load_model(path):

    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print("Error",e)

def run_load_model():

    # load plate detection model>>
    
    wpod_net_path = "model/wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    # Load model architecture, weight and labels
    json_file = open('model/MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model/License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('model/license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")

    return wpod_net,model,labels

def preprocess_image(image_path,resize=False):

    img = cv2.imread(image_path)
    # img = image_path
    # img = cv2.convertScaleAbs(img, alpha=0.5, beta=100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, vehicle_type, Dmax=608, Dmin=440):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type, cor, plate_ratio = detect_lp(wpod_net, vehicle, bound_dim,vehicle_type, lp_threshold=0.9)
    return vehicle, LpImg,lp_type, cor, plate_ratio

def sort_contours(cnts,reverse = False):
    
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts,boundingBoxes

def getBinaryImage(image_array):

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY_INV +  cv2.THRESH_OTSU)[1]

    return binary

def prep_image(image_path,vehicle_type):

    # Obtain plate image and its coordinates from an image
    test_image = image_path
    _, LpImg,lp_type,_, plate_ratio = get_plate(test_image,vehicle_type)
    # print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])

    if (len(LpImg)):
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        cv2.imwrite("OUTPUT/plate_id_image/original_plate.png",plate_image)
        if lp_type == 1: plate_image = plate_image[15:plate_image.shape[0] - 17, 10:plate_image.shape[1]-15]
        else:plate_image = plate_image[10:plate_image.shape[0] - 37, 3:plate_image.shape[1]-3]
        # convert to grayscale and blur the image
        binary = getBinaryImage(plate_image)
        cv2.imwrite("OUTPUT/plate_id_image/binary_plate.png",binary)
        # check to find contour more better for sementation.

        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cont,binary,plate_image,lp_type,plate_ratio
    
def predict_from_model(image,model,labels):

    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def drawKhmer_cont(test_roi,x_col,y_col,lp_type):
    try:
        if lp_type == 2:
            y_min = min(y_col)
            cv2.rectangle(test_roi, (50, 0), (test_roi.shape[1]-50, y_min-1), (0, 255,0), 2)
            khmer_org_crop = test_roi[0:y_min-1, 50:test_roi.shape[1]-50]
        if lp_type == 1: 
            x_min = min(x_col)
            cv2.rectangle(test_roi, (10, 10), (x_min-10,test_roi.shape[0]-20), (0, 255,0), 2)
            khmer_org_crop = test_roi[10:test_roi.shape[0]-20, 10:x_min-10]

        khmer_org_crop = getBinaryImage(khmer_org_crop)
        cv2.imwrite("OUTPUT/khmer_segment/khmer_crop.png",khmer_org_crop)
            
        return khmer_org_crop
    except:pass

def detection_char(cont,binary,plate_image,lp_type,display = False):

    # create a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    x_col, y_col = [],[]
    cont, _ = sort_contours(cont)

    crop_characters = []
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0,0), 1)
        if lp_type == 2:
            ratio = h/w
            if 1<=ratio<=6 and 0.37<=h/plate_image.shape[0]<=0.9: # Only select contour with defined ratio
                if 0.17<=y/plate_image.shape[1]<=1 and 0<=x/plate_image.shape[1]<=1: # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                    # Seperrate number and gibe prediction
                    y_col.append(y)
                    y_col.append(y+h)
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        if lp_type == 1:
            ratio = h/w
            if 1<=ratio<=6 and plate_image.shape[0]-(y+h)<=15:  # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.40 and x/plate_image.shape[1]>=0.32: # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                    # Seperrate number and gibe prediction
                    x_col.append(x)
                    x_col.append(x+w)
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

    khmer_org_crop = drawKhmer_cont(test_roi,x_col,y_col,lp_type)

    cv2.imwrite("OUTPUT/plate_id_image/plate_id.png",test_roi)
    if display:
        fig = plt.figure(figsize=(10,6))
        plt.imshow(test_roi)
        plt.show()
    return crop_characters,khmer_org_crop

def recognition_char(crop_characters):

    final_string = ''
    for character in crop_characters:
        # fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character,model,labels))
        final_string+=title.strip("'[]")

    return final_string

def final_result_func(final_result,predicted_result,khmerText,centroid_w):

    info_dict = {"plate_id":[predicted_result],
                "khmer_org":[khmerText],
                "centroid":[centroid_w]
                }
    if final_result: 
        flag = False
        for plate_ref in final_result:
            if SequenceMatcher(a=predicted_result,b=final_result[plate_ref]["plate_id"][0]).ratio() >= 0.7:
                flag = True
                final_result[plate_ref]["plate_id"].append(predicted_result)
                final_result[plate_ref]["khmer_org"].append(khmerText)
                final_result[plate_ref]["centroid"].append(centroid_w)
        if flag is False:
            final_result[f"{len(final_result)}"] = info_dict
    else : final_result["0"] = info_dict
    return final_result

def create_dataset(crop_characters,plate_id):

    plate_id_path = f"TRAINING_IMAGE/dataset_image_training/{plate_id}"
    if len(plate_id)>=5:
        try:
            os.mkdir(f"{plate_id_path}") 
            for i,segment_char in enumerate(crop_characters):
                cv2.imwrite(f'{plate_id_path}/{plate_id[i]}_{i}.png',segment_char)
        except:
            os.mkdir("TRAINING_IMAGE/dataset_image_training") 
            pass
        finally:
            return plate_id_path

def start_engine_image(image_path,vehicle_type,getDataset = False):

     # Initialize a list which will be used to append charater images
    cont,binary,plate_image,lp_type,plate_ratio = prep_image(image_path,vehicle_type)
    crop_characters,khmer_org_crop = detection_char(cont,binary,plate_image,lp_type,False)
    plate_id = recognition_char(crop_characters)
    if getDataset: 
        plate_id_path = create_dataset(crop_characters,plate_id)
        return plate_id,plate_image,plate_id_path,plate_ratio,lp_type
    else: return plate_id,plate_ratio
    
wpod_net,model,labels  = run_load_model()

# if __name__ == "__main__":

    # FOR VIDEO TESTING>>>>

    # q = queue.Queue()
    # # Receive()
    # p1 = threading.Thread(target=Receive)
    # p2 = threading.Thread(target=Display)
    # p1.start()
    # p2.start()
    

    # FOR IMAGE TESTING>>>>>

    
   