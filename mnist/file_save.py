import json

def change_json(x,y,z,vector_list):
    vector_list.append({'vector':[x,y,z]})
    return(vector_list)

def get_vec(X_reduce, vector_list):
    for line in X_reduce:
        x = line[0]
        y = line[1]
        z = line[2]
        vector_list=change_json(x,y,z,vector_list)
    return(vector_list)

def file_save(vector_list):
    fw = open('tsne.json','w')
    json.dump(vector_list,fw,indent=4)

def main(X_reduce):
    vector_list=[]
    vector_list = get_vec(X_reduce, vector_list)
    file_save(vector_list)
    
