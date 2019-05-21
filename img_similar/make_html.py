import pickle

def main():
    path = 'img.html'
    img_number= 10
    uncorrect_img = 9
    with open('errlabel_list.txt', 'rb') as list_result:
        data = pickle.load(list_result)
        with open(path, mode='a') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<head>\n')
            f.write('</head>\n')
            f.write('<body>\n')
            for i, dec in enumerate(data):
                f.write('<H2>uncorrect image, predict_label=%s, true_label=%s</H2>'%(dec['predict_label'], dec['true_label']))
                f.write('<img src='+'\'uncorrect_img/%03.f'%(i)+'.png\'>\n')
                for k in range(img_number):
                    f.write('<img src='+'\'similar_img/%03.f'%(k)+'.png\'>\n')
        f.write('</body>\n')
        f.write('</html>\n')
    f.close()

if __name__ == '__main__':
    main()
