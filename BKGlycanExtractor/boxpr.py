#!/bin/env python3.12

from pylab import *
import random
import math
imgw = 800
imgh = 600

def dist(boxes,point):
    mindist = 1e+20
    for b in boxes:
        dx = abs(b['cx']-point[0])
        dy = abs(b['cy']-point[1])
        if dx < mindist:
            mindist = dx
        elif dy < mindist:
            mindist = dy
    return mindist

def intersect(b1,b2):
    if b1['x'] <= b2['x'] <= b1['x']+b1['w'] and \
       b1['y'] <= b2['y'] <= b1['y']+b1['y']:
        return True
    if b1['x'] <= b2['x']+b2['w'] <= b1['x']+b1['w'] and \
       b1['y'] <= b2['y'] <= b1['y']+b1['y']:
        return True
    if b1['x'] <= b2['x'] <= b1['x']+b1['w'] and \
       b1['y'] <= b2['y']+b2['h'] <= b1['y']+b1['y']:
        return True
    if b1['x'] <= b2['x']+b2['w'] <= b1['x']+b1['w'] and \
       b1['y'] <= b2['y']+b2['h'] <= b1['y']+b1['y']:
        return True
    if b2['x'] <= b1['x'] <= b2['x']+b2['w'] and \
       b2['y'] <= b1['y'] <= b2['y']+b2['y']:
        return True
    if b2['x'] <= b1['x']+b1['w'] <= b2['x']+b2['w'] and \
       b2['y'] <= b1['y'] <= b2['y']+b2['y']:
        return True
    if b2['x'] <= b1['x'] <= b2['x']+b2['w'] and \
       b2['y'] <= b1['y']+b1['h'] <= b2['y']+b2['y']:
        return True
    if b2['x'] <= b1['x']+b1['w'] <= b2['x']+b2['w'] and \
       b2['y'] <= b1['y']+b1['h'] <= b2['y']+b2['y']:
        return True
    return False

def intersection(b1,b2):
    assert intersect(b1,b2)
    ix=max(b1['x'],b2['x'])
    iy=max(b1['y'],b2['y'])
    iw=max(min(b1['x']+b1['w'],b2['x']+b2['w'])-ix,0)
    ih=max(min(b1['y']+b1['h'],b2['y']+b2['h'])-iy,0)
    return dict(cx=ix+iw//2,cy=iy+ih//2,x=ix,y=iy,w=iw,h=ih)

def area(b):
    return (b['w']+1)*(b['h']+1)

def iou(b1,b2):
    if intersect(b1,b2):
        i = intersection(b1,b2)
        return float(area(i))/(area(b1)+area(b2)-area(i))
    return 0

def assignboxes(tboxes,dboxes,edges):
    # precision for confidence thresholds, critical values
    cps = set([ math.floor(100*e['conf'])/100 for e in edges ])
    retval = []
    for cp in sorted(cps):
        tassigned = set()
        dassigned = set()
        assignment = []
        tp=0; fp=0; fn=0
        for e in sorted(edges,key=lambda e: (-e['conf'],-e['iou'])):
            if e['conf'] < cp:
                break
            if e['tbox'] in tassigned:
                continue
            if e['dbox'] in dassigned:
                continue
            tassigned.add(e['tbox'])
            dassigned.add(e['dbox'])
            if e['cls'][0] == e['cls'][1]:
                tp += 1
            else:
                fp += 1
                fn += 1
        fn += len(tboxes)-len(tassigned)
        fp += len([ d for d in dboxes if d['conf'] >= cp])-len(dassigned)
        retval.append((cp,tp,fp,fn))
    return retval

size = 22
nbox = random.randint(10,20)
tboxes = []
dboxes = []
while True:
    cx = random.randint(0+size,imgw-size)
    cy = random.randint(0+size,imgh-size)
    cls = random.randint(0,5)
    if dist(tboxes,(cx,cy)) > size:
        tboxes.append(dict(cx=cx,cy=cy,x=cx-size//2,y=cy-size//2,w=size,h=size,cls=cls))
        if random.random() > 0.1:
            dx = random.randint(-5,5)
            dy = random.randint(-5,5)
            dsx = random.randint(-3,3)
            dsy = random.randint(-3,3)
            dcls = 1*(random.random()>0.9)
            dboxes.append(dict(conf=(0.9+random.random()*0.1),cx=cx+dx,cy=cy+dy,
                           x=cx+dx-((size+dsx)//2),y=cy+dy-((size+dsy)//2),
                           w=(size+dsx),h=(size+dsy),cls=cls+dcls))
        if len(tboxes) >= nbox:
            break

# print(tboxes)
# print(dboxes)

# could move this into the assign function, perhaps
edges = []
for i,tbox in enumerate(tboxes):
    for j,dbox in enumerate(dboxes):
        iouij = iou(tbox,dbox)
        # if iouij > 0:
        #     print(i,tbox['x'],tbox['y'],tbox['x']+tbox['w'],tbox['y']+tbox['h'])
        #     print(j,dbox['x'],dbox['y'],dbox['x']+dbox['w'],dbox['y']+dbox['h'])
        #     print(iouij)
        #
        # consider box pairs to determine whether they should be potential matches
        #
        if iouij >= 0.5:
            edges.append(dict(tbox=i,dbox=j,iou=iouij,conf=dbox['conf'],cls=(tbox['cls'],dbox['cls'])))

# print(len(tboxes))
# print(len(dboxes))
# print(len(edges),edges)

# Check tp,fp,fn are monotonic with confidence thresholds...
x = []; y = []
for cp,tp,fp,fn in sorted(assignboxes(tboxes,dboxes,edges)):
    print("conf=",cp,"tp=",tp,"fp=",fp,"fn=",fn,"prec=",float(tp)/(tp+fp),"recall=",float(tp)/(tp+fn))
    # recall
    x.append(float(tp)/(tp+fn))
    # precision
    y.append(float(tp)/(tp+fp))

# remove non-monotonic values...
x1 = []; y1 = []
for i in range(len(x)):
    if len(x1) == 0:
        x1.append(x[i])
        y1.append(y[i])
    elif y[i] > y1[-1]:
        x1.append(x[i])
        y1.append(y[i])

# and make step-based...
x2 = []; y2 = []
x2.append(x1[0])
y2.append(0)
x2.append(x1[0])
y2.append(y1[0])
for i in range(1,len(x1)):
    x2.append(x1[i])
    y2.append(y1[i-1])
    x2.append(x1[i])
    y2.append(y1[i])
x2.append(0)
y2.append(y2[-1])

plot(x2,y2,'-')
plot(x,y,'r.')
xlim([0,1.1])
ylim([0,1.1])
ylabel('precision')
xlabel('recall')
show()
