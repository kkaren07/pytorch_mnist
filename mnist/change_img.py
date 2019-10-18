from PIL import Image
import json

def save_img(img,num):
    img2 = img.resize((140, 140))
    filename = 'tsne_mnist_img/img_{0}.jpg'.format(num)
    img2.save(filename)
    
def change_img(img_binary):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(img_binary[i+j*28]*256)
    return(img)

def main():
    f = open("images.json", 'r')
    json_data = json.load(f)
    num=0
    
    for index in range(len(json_data)):
        img = change_img(json_data[index]['img'])
        num = index
        save_img(img,num)

if __name__ == '__main__':
    main()
        
    
