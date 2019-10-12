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
 
html='''
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="test1.css">
    </head>
    <body>
'''
css = '' 
for key,value in div.items():
    match = similarity(value[0],'Header')
    if match > 0.7:
        html+='<div id="'+value[0]+'">'
        html+=value[0]
        html+='</div><br>'
        css+='#'+value[0]+'{color:white;position: fixed;top:0%;width: 100%;background-color:#606060;height:'+value[1]+'}'
    match = similarity(value[-1],'TEXT')
    if  match > 0.7:
        html+=value[0]+': '
        html+='<input type="text" id="'+value[0]+'">'
        html+='<br>'
        css+='#'+value[0]+'{margin:'+value[1]+';}'
    match = similarity(value[-1],'Button')
    if match > 0.7:
        html+=value[0]+': '
        html+='<button type="button" id="'+value[0]+'">'+value[0]+'</button>'
        html+='<br>'
        css+='#'+value[0]+'{margin:'+value[1]+';}'
    match = similarity(value[-1],'IMAGE')
    if match > 0.7:
        html+='<img src="1.jpg" alt="'+value[0]+'">'
        html+='<br>'
        css+='img{;width:304px;height:228px}'
        
html+="</body></html>"
with open('test1.html','w') as fopen:
    fopen.write(html)
with open('test1.css','w') as fopen:
    fopen.write(css)