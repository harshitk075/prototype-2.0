from PIL import Image
import pytesseract

import cv2
import os
img_path = "test.png"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
cv2.waitKey(0)


from difflib import SequenceMatcher
def similarity(a,b):
    return SequenceMatcher(None,a,b).ratio()

from collections import defaultdict
div=defaultdict(list)
ans = text.split('\n')
for i in ans:
    val = i[-1:]
    div[val].append(i.split('#')[0].strip())
div.pop('',None)
 
js = '' 
for key,value in div.items():
    match = similarity(value[-1],'TEXT')
    if  match > 0.7:
        js+='var t=document.createTextNode("'+value[0]+'");' \
        +'document.body.appendChild(t);' \
        +'var x = document.createElement("INPUT");' \
        +'x.setAttribute("type", "text");' \
        +'x.setAttribute("style", "margin:'+value[1]+'");' \
        +'document.body.appendChild(x);' \
        +'document.write("<br>");'
    match = similarity(value[-1],'Button')
    if match > 0.7:
        btn='var btn = document.createElement("BUTTON");' \
        +'var t = document.createTextNode("'+value[0]+'");' \
        + 'btn.appendChild(t);' \
        +'document.body.appendChild(btn);' \
        + 'btn.setAttribute("style", "margin:'+value[1]+'");' \
        +'document.write("<br>");'
        js+=btn
    match = similarity(value[-1],'IMAGE')
    if match > 0.7:
        img='var x = document.createElement("IMG");' \
        +'x.setAttribute("src", "test.png");' \
        +'x.setAttribute("width", "304");' \
        +'x.setAttribute("height", "228");' \
        +'x.setAttribute("alt", "image");' \
        +'document.body.appendChild(x);'
        js+=img
    match = similarity(value[0],'Header')
    if match > 0.7:
        header='var x = document.createElement("DIV");'\
        +'var t = document.createTextNode("Header");'\
        +'x.setAttribute("style", "background-color: #606060;height:20px;margin:0px;");' \
        +'x.setAttribute("style", "margin-top:0%");'
        js+=header
    match = similarity(value[0],'Footer')
    if match > 0.7:
        footer='var x = document.createElement("Footer");'\
        +'var t = document.createTextNode("Footer");'\
        +'x.setAttribute("style", "height:'+value[1]+'");' \
        +'x.setAttribute("style", "margin-bottom:0%");'
        js+=footer
with open('test1.js','w') as fopen:
    fopen.write(js)