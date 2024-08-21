import sys
import os
import re
import math

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

dbox_file = open("./out/prc_log.txt").read().split('\n')
dboxes = {}
pos = -1
img_name = ''
model_name = ''
name = ''
for line in dbox_file:
    pos += 1
    if line.find("Start") >= 1:
        img_name = line.split('.')[0]
        continue
    if line.find("model") >= 0:
        model_name = line
        continue
    if line.find("Finished") >= 0:
        img_name = ''
        name = ''
        continue
    if line.find("Confidence:") >= 0:
        name = img_name + '_' +model_name + '_' + line
        dboxes[name] = []
        continue
    if line.startswith("Detected") and len(name) > 0:
        for i in dbox_file[pos+1:-1:2]:
            if i.startswith("["):
                i = re.sub('[^0-9:]','',i)
                cords = re.split('.:',i)[1:]
                if not cords in dboxes[name]:
                    dboxes[name].append(cords)
            else:
                break


monos = {}
reallinks = {}
for dirpath,dirnames,filenames in os.walk('./data'):
    for name in filenames:
        if name.find("_map.txt") >= 0:
            map_name = name.split('_')[0]
            map_file = open("./data/"+name).read().split('\n')
            monos[map_name] = []
            reallinks[map_name] = []
            for l in map_file:
                if l.startswith('m'):
                    monocenter = l.split('\t')[-2].split(',')
                    monoid = l.split('\t')[1]
                    monotype = l.split('\t')[2]                  
                    monos[map_name].append([monoid, monotype, monocenter])
                if l.startswith('l'):
                    reallinks[map_name].append(l.split('\t')[1:])

links = {}
for predict in dboxes:
    links[predict] = []
    for dbox in dboxes[predict]:
        monoin = []
        x0 = int(dbox[0])
        x1 = int(dbox[1])
        y0 = int(dbox[2])
        y1 = int(dbox[3])
        for mono in monos[predict.split('_')[0]]:
            xc = int(mono[2][0])
            yc = int(mono[2][1])
            if xc > x0 and xc < x1 and yc > y0 and yc < y1:
                monoin.append(mono)
        if len(monoin) == 2:
            r = [monoin[0][0],monoin[1][0]]
            links[predict].append(r)
        if len(monoin) > 2 and len(monoin) < 4:
            dis = 0
            r = []
            for m in monoin:
                for n in monoin:
                    if math.fabs(int(m[2][0])-int(n[2][0]))+math.fabs(int(m[2][1])-int(n[2][1])) > dis:
                        dis = math.fabs(int(m[2][0])-int(n[2][0]))+math.fabs(int(m[2][1])-int(n[2][1]))      
                        r = [m[0],n[0]]
            if not r in links[predict] and len(r) > 0:
                links[predict].append(r)
tpr = {}  
for predict in links:
    if float(predict.split('_')[2].split(' ')[1]) < 0.918: 
        print(predict,':')
    if not predict.split('_')[1] in tpr:
        tpr[predict.split('_')[1]] = {}
    if not predict.split(':')[1] in tpr[predict.split('_')[1]]:
        tpr[predict.split('_')[1]][predict.split(':')[1]] = ''

    for i in reallinks[predict.split('_')[0]]:
        if i in links[predict] or [i[1],i[0]] in links[predict]:
            tpr[predict.split('_')[1]][predict.split(':')[1]] += 'TP.'
            if float(predict.split('_')[2].split(' ')[1]) < 0.918:
                print(i,':','TP')
        else :
            tpr[predict.split('_')[1]][predict.split(':')[1]] += 'FN.'
            if float(predict.split('_')[2].split(' ')[1]) < 0.918:
                print(i,':','FN')
    for n in links[predict]:
        if not n in reallinks[predict.split('_')[0]] and not [n[1],n[0]] in reallinks[predict.split('_')[0]]:
            tpr[predict.split('_')[1]][predict.split(':')[1]] += 'FP.'
            if float(predict.split('_')[2].split(' ')[1]) < 0.918:
                print(n,':','FP')

#print(tpr)
def plotprecisionrecall(tpr):
    for model in tpr:
        precision = []
        recall = []
        for conf in tpr[model]:          
            results_list = tpr[model][conf]
            #print(results_list)
            fp = results_list.count('FP')
            tp = results_list.count('TP')
            pos = fp + tp
            fn = results_list.count('FN')
            tpfn = tp + fn
            try: 
                prec = tp/pos
            except ZeroDivisionError:
                prec = 0
            rec = tp/tpfn
            precision.append(prec)
            recall.append(rec)
            #print(rec)
            #print(prec)
        recall, precision = zip(*sorted(zip(recall, precision)))
        if len(set(recall)) == 1 and len(set(precision)) == 1:
            plt.plot(recall,precision, ".", label = model)
        else:
            plt.plot(recall,precision, ".-",label = model)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0.8,1.1])
    plt.ylim([0.8,1.1])
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(loc="best")
    pr = plt.gcf()
    return pr

                
#make plot    
prc = plotprecisionrecall(tpr)    


#save plot
path = f"./precisionrecallplot.png"
if os.path.isfile(path):
    path = f"./precisionrecallplot"+".png"
    print("Directory already contains prc curve file - check")
plt.savefig(path)
plt.close()
            
