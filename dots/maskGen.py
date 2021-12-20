import os
import cv2
import json
import numpy as np 

source_image_folder = os.path.join(os.getcwd(), "train", "image")
json_path = os.path.join(os.getcwd(), "dots", "maskGen_json.json")                  # Relative to root directory
count = 40                                           # Count of total images saved
file_bbs = {}                                       # Dictionary containing polygon coordinates for mask
MASK_WIDTH = 640				    # Dimensions should match those of ground truth image
MASK_HEIGHT = 535								

# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["y"]
        
        width = data[itr]["regions"][count]["shape_attributes"]["width"]
        height = data[itr]["regions"][count]["shape_attributes"]["height"]

        print("Points extracted", key)

    except:
        print("No BB. Skipping", key)
        return
    
    all_points = []
    all_points.append([x_points, y_points, width, height])
    file_bbs[key] = all_points

# Read JSON file
with open(json_path) as f:
  data = json.load(f)

for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 1               # Contains count of masks for a single ground truth image

    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count+1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nDict size: ", len(file_bbs))
        
# For each entry in dictionary, generate mask and save in correponding 
# folder
for itr in file_bbs:
    num_masks = itr.split("*")
    mask_folder = os.path.join(os.getcwd(), "dots", "train", "label")
    mask = np.ones((MASK_HEIGHT, MASK_WIDTH))
    arr = np.zeros(4)
    #try:
    arr[0] = int(file_bbs[itr][0][0])
    arr[1] = int(file_bbs[itr][0][1])
    arr[2] = int(file_bbs[itr][0][0] + file_bbs[itr][0][2])
    arr[3] = int(file_bbs[itr][0][1] + file_bbs[itr][0][3])
    count += 1
    pt1 = (int(arr[0]), int(arr[1]))
    pt2 = (int(arr[2]), int(arr[3]))

    VALUE_TO_FILL = 255
    cv2.rectangle(mask, pt1, pt2, (VALUE_TO_FILL), -1)
    
    if len(num_masks) > 1:
        cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".png") , mask)    
    else:
        cv2.imwrite(os.path.join(mask_folder, itr + ".png") , mask)
        
print("Images saved:", count)