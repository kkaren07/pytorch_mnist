import json
import numpy

def change_json(x,y,z,vector_list):
    vector_list.append({'vector':[x,y,z]})
    return(vector_list)

def get_vec(X_reduce, vector_list):
    for line in X_reduce:
        x = float(line[0])
        y = float(line[1])
        z = float(line[2])
        vector_list=change_json(x,y,z,vector_list)
    return(vector_list)

def file_save(vector_list):
    fw = open('tsne.json','w')
    json.dump(vector_list,fw,indent=4)

def main_vec(X_reduce):
    vector_list=[]
    vector_list = get_vec(X_reduce, vector_list)
    file_save(vector_list)

def change_json_img(img,images):
    images.append({'img':img})
    return(images)
    
def get_img(img_list, images):
    for img in img_list:
        img=img.tolist()
        images=change_json_img(img,images)
    return(images)
        
def file_img_save(img_list):
    f_img = open('images.json','w')
    json.dump(img_list,f_img,indent=4)
    
def main_img(img_list):
    images=[]
    images=get_img(img_list, images)
    file_img_save(images)
    
