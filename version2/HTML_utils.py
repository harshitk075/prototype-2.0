import numpy as numpy
import pandas as pd 
from OCR_utils import remove_low_prob_ocr

def tesseract_output_formatter(output_df, threshold):
    """Removes low probability OCR predictions and converts tesseract_ocr's output dataframe to a list of size[n x 5]"""
    ocr_bboxes = remove_low_prob_ocr(output_df, threshold)
    ocr_bboxes = output_df[['left','top','width','height','text']].copy(deep=True)
    ocr_bboxes['width'] += ocr_bboxes['left']
    ocr_bboxes['height'] += ocr_bboxes['top']
    return ocr_bboxes.values

def split_into_rows(units):
    units = units.groupby(1)
    rows = [row for _,row in units]
    return rows

def split_into_columns(row):
    row = row.groupby(0)
    columns = [column.values.tolist() for _,column in row]
    return (columns)

def quantizer(units, q):
    """Quantizes all units' coordinates and sorts them into rows and columns.
        Output grid is a non-uniform list since each row may have different no. of columns
        ex: grid[3][4][0] gives the [xmin, ymin, xmax, ymax, class] list of element in
            3rd row, 4th column
    """
    units = pd.DataFrame(units)
    units[[0,1,2,3]] = (units[[0,1,2,3]]/q).astype('int64') *q #Quantization
    rows = split_into_rows(units)
    grid = [split_into_columns(row) for row in rows]
    return (grid)    

import numpy as np
import pandas as pd

def row_quantizer(units, q):
    """Quantizes all units' coordinates and sorts them into rows and columns.
        Output grid is a non-uniform list since each row may have different no. of columns
        ex: grid[3][4][0] gives the [xmin, ymin, xmax, ymax, class] list of element in
            3rd row, 4th column
    """
    #units = units.values
    units = pd.DataFrame(units)
    units[[0,1,2,3]] = (units[[0,1,2,3]]/q).astype('int64')*q #Only row quantization
    rows = split_into_rows(units)
    #grid = [split_into_columns(row) for row in rows]
    return [row.sort_values(by = [0]).values.tolist() for row in rows]

def getHTML(component):
    return "<!DOCTYPE html>\n\n<html lang=\"en\">\n<head>\n    <meta name=\"viewport\" content=\"width=device-width\" />\n    <title>HTML Result</title>\n    <link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css\"\n\n          integrity=\"sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB\" crossorigin=\"anonymous\">\n</head>\n<body>\n    <div class=\"container body-content\">\n<div class=\"container\">\n" + component + "</div>\n\n    </div>\n</body>\n</html>"

def getRow(components):
    code = "<div class=\"row justify-content-start\" style=\"padding-top:10px;\">\n"
    code = code + components
    code = code + "</div>\n"
    return code

def getCol(component):
    return "<div class=\"col\" style=\"padding-top:10px;\">\n" + component + "</div>\n"

def getLabel(text):
    return '<label>{}</label>\n'.format(text)

def getButton(text):
    return "<button class=\"btn btn-primary\">" + text + "</button>\n"

def getLink(text):
    return "<a href="">" + text + "</a>\n"

def getHeading(text):
    return "<h1>" + text + "</h1>\n"

def getParagraph():
    return "<p class=\"text-black-50\">\n    Lorem ipsum dolor sit amet, consectetur adipiscing elit\n    <br />\n    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n    <br />\n    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris\n</p>\n"

def getImage():
    return "<img alt=\"Image html\" width=\"90%\" height=\"90%\" style=\"max-height:500px;max-width:500px;\" src=\"https://sketch2code.azurewebsites.net/Content/img/fake_img.svg\" />"

def getComboBox(text):
    return "<div class=\"dropdown\">\n<button type=\"button\" class=\"btn btn-primary dropdown-toggle\" data-toggle=\"dropdown\">\n" + text + "\n</button>\n<div class=\"dropdown-menu\">\n<a class=\"dropdown-item\" href=\"#\"></a>\n\n</div>\n</div>\n"

def getCheckBox(text):
    return "<label><input type=\"checkbox\" />"+ text + "</label>\n"

def getRadioButton(text):
    return "<label><input type=\"radio\" />"+ text +"</label>\n"

def getTextBox():
    return "<input class=\"form-control\"></input>\n"

def getElement(c, arg1='Lorem ipsum'):
    arg1 = str(arg1)
    #print(c,arg1)
    if(c==1):
        return getRadioButton(arg1)
    elif (c==2):
        return getHeading(arg1)
    elif (c==3):
        return getLink(arg1)
    elif (c==4):
        return getTextBox()
    elif (c==5):
        return getCheckBox(arg1)
    elif (c==6):
        return getLabel(arg1)
    elif (c==7):
        return getImage()
    elif (c==8):
        return getButton(arg1)
    elif (c==9):
        return getParagraph()
    elif (c==10):
        return getComboBox(arg1)

def generate_html(lbl_bboxes,tesseract_output):
    """Make sure lbl_bboxes -> xmin,ymin,xmax,ymax"""
    doc = ''
    
    ocr_bboxes = tesseract_output_formatter(tesseract_output, 0)
    units = np.append(lbl_bboxes,ocr_bboxes,axis=0)
    
    grid = quantizer(units,0.1)
    ocr_rows  = row_quantizer(ocr_bboxes,0.1)

    for row in grid:
        row_doc = ''
        for column in row:
            if column[0][4] in range(1,11):
                if (len(column) == 1):
                    output = (getElement(column[0][4]))
                else:
                    text = "Lorem ipsum"
                    c    = column[0][4]
                    #search for the string in column[1][4] in ocr_grid
                    for row in ocr_rows:
                        if column[1] in row:
                            words = []
                            for word in row:
                                if (word[0]<=column[0][2] and word[0]>=column[0][0] and isinstance(word[4],str)):
                                    # print(type(word[4]))
                                    words.append(word[4])
                            text = ' '.join(words)
                            break
                    #In that row, concat all strings in order whose y_min < ymax of the element bbox

                    output = (getElement(c,text))

                if (len(row)==1):
                    row_doc += output
                else:
                    row_doc += getCol(output)
        doc += getRow(row_doc)
    doc = getHTML(doc)     
    return doc