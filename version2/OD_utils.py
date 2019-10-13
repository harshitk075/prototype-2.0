import numpy as np

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def merge(boxes,classes):
    return np.append(boxes,classes,axis=1)

def remove_low_prob(boxes, scores, prob_threshold):
    return np.delete(boxes,np.where(scores<prob_threshold)[0],0)

def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []
 
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	pick = []
 
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	area = (x2 - x1) * (y2 - y1)
	idxs = np.argsort(y2)
 
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		w = np.maximum(0, xx2 - xx1)
		h = np.maximum(0, yy2 - yy1)
 
		overlap = (w * h) / area[idxs[:last]]
 
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	return boxes[pick]