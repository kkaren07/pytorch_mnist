import pickle

def main():
    path = 'img.html'
    n = 0
    img_number = 10
    uncorrect_img = 9
    with open('err_list.txt', 'rb') as list_result:
        data = pickle.load(list_result)
    with open('judge_label.txt', 'rb') as judge:
        judge_label = pickle.load(judge)
    with open(path, mode='a') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<meta charset=\'utf-8\'>')
        f.write('<link rel=\'stylesheet\' href=\'stylesheet.css\'>')
        f.write('</head>\n')
        f.write('<body>\n')
        for (i, dec) in enumerate(data):
            f.write('<h2>test data with uncorrect label</h2>\n')
            f.write('<p>predict_label=%s, true_label=%s</p>'%(dec['predict_label'], dec['true_label']))
            f.write('<div class=\'noise_item\'>\n')
            f.write('<img src='+'\'uncorrect_img/%03.f'%(i)+'.png\'>\n')
            f.write('</div>\n')
            f.write('<div class=\'clearfix\'>\n')
            for k in range(n, img_number):
                if judge_label[k]['from_load_data.py']!=judge_label[k]['from_trainset']:
                    f.write('<p>train data with uncorrect label\n')
                    f.write('</p>\n')
                    f.write('<p>noise_label=%s, true_label=%s'%(judge_label[i]['from_load_data.py'], judge_label[i]['from_trainset']))
                    f.write('</p>')
                f.write('<div class=\'similar_item\'>\n')
                f.write('<img src='+'\'similar_img/%03.f'%(k)+'.png\'>\n')
                f.write('</div>\n')
            f.write('</div>\n')#clearfixのdiv
            n=n+10
            img_number=img_number+10
        f.write('</body>\n')
        f.write('</html>\n')
   
if __name__ == '__main__':
    main()
