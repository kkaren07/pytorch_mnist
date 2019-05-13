path = 'img.html'
img_number= 10
uncorrect_img = 22
with open(path, mode='a') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<head>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    for i in range(uncorrect_img):
        f.write('<H2>uncorrect image</H2>')
        f.write('<img src='+'\'uncorrect_img/%03.f'%(i)+'.png\'>\n')
        for k in range(img_number):
            f.write('<img src='+'\'similar_img/%03.f'%(k)+'.png\'>\n')
       
    f.write('</body>\n')
    f.write('</html>\n')
f.close()
