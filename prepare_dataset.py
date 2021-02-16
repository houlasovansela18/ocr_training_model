import os
import cv2
import shutil

def plateImg_to_folder(directory):

    import main as main_recognition
    k = 0
    cor_length_ratio_dict = {
        "1": [],
        "2": []
    }
    wro_legnth_ratio_dict = {
        "1": [],
        "2": []
    }
    try:
        for i,filename in enumerate(os.listdir(directory)):
            if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg"): 
                    image_path = f"{directory}/{filename}"
                    plate_id,plate_image,plate_id_path,plate_ratio,lp_type = main_recognition.start_engine_image(image_path,"car",getDataset = True)
                    # print(plate_id)
                    plate_ratio = "%.2f" %plate_ratio
                    if len(plate_id)>=5:
                        cor_length_ratio_dict[f"{lp_type}"].append(plate_ratio)
                        cv2.imwrite(f"{plate_id_path}/{plate_id}_{plate_ratio}.png" ,plate_image)
                        print(f"[{k+1}] Success loaded | produce output  images out of {len(os.listdir(directory))} images...>>")
                        k+=1
                        os.remove(image_path)
                    else: 
                        wro_legnth_ratio_dict[f"{lp_type}"].append(plate_ratio)
            else:
                continue
        min_ratio = round(float((float(min(cor_length_ratio_dict["1"]))+float(max(cor_length_ratio_dict["2"])))/2))
        print(f"Wrong plate_ratio dict :{wro_legnth_ratio_dict}")
        print(f'Best plate_ratio:{min_ratio} of {len(cor_length_ratio_dict["1"])+len(cor_length_ratio_dict["2"])} plate_image')
    except Exception as e:
        print(e)
        pass

def easyName_dataset(dataset_dir):

    try:
        if os.path.isfile(f"{dataset_dir}/.DS_Store"): os.remove(f"{dataset_dir}/.DS_Store")
        total_image = 0
        for foldername in sorted(os.listdir(dataset_dir)):
            k = 0
            for i,filename in enumerate(sorted(os.listdir(f"{dataset_dir}/{foldername}"))):
                if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg"): 
                    k+=1
                    oldfile = f"{dataset_dir}/{foldername}/{filename}"
                    newfile = f'{dataset_dir}/{foldername}/{foldername}_{k}.jpg'
                    # print(oldfile,newfile)
                    os.rename (oldfile,newfile)
            total_image+=k
            print(f"[{k}] Images successfully rename in folder : [{foldername}]")
        print(f"[{total_image}] Images successfully rename")
    except Exception as e:
        print(e)
        pass
        
def segmentImg_to_dataset(input_folder,wrong_char,right_char):

    tail_name = ""
    for foldername in os.listdir(f"{dataset_dir}"):
                if foldername == right_char: 
                    tail_name = len(os.listdir(f"{dataset_dir}/{foldername}"))+1
    src = f"{defaultImg_path}/{input_folder}/{wrong_char}.png"
    dst = f"{dataset_dir}/{right_char}/{right_char}_{tail_name}.png"

    shutil.copyfile(src, dst)
    print(f"Successfully copy image [{scr}] to [{dst}]")

if __name__ == "__main__":

    # [ DO NOT DELETE THIS PATH OTHERWISE ASSERT PATH ERROR ]
    defaultImg_path = "TRAINING_IMAGE/dataset_image_training"
    source_dir = "TRAINING_IMAGE/Vehicle_plate_image"
    dataset_dir = "TRAINING_IMAGE/dataset_characters"
    # -----------------------------------------------

    # INPUT FOLDER FOR MAPPING DATASET
    # plate_folder = "XXXXXXX"
    plate_folder = "2V4113"
    # wrong_char = "[a-z],[A-Z]" --> wrong segmentImage PATH 
    wrong_char = "1_3"
    # right_char = "[a-z],[A-Z]" --> correct string compare with real image 
    right_char = "7"

    # -----------------------------------------------

    # FUNCTION CALLING

    # for add wrong image to dataset folder
    # segmentImg_to_dataset(plate_folder,wrong_char.upper(),right_char.upper())

    # for gather plate_id image and segment image
    plateImg_to_folder(source_dir)

    # for rename dataset folder
    # easyName_dataset(dataset_dir)
    # -----------------------------------------------  