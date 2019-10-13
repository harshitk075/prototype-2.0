import numpy as np
import os
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from PIL import Image
from HTML_utils import *
from OCR_utils import *
from OD_utils import *
from selenium import webdriver
import time

# This is needed since the notebook is stored in the object_detection folder.
root = os.getcwd()
sys.path.append(os.path.join(root,'models','research'))

from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util



# Path of model to be loaded.
MODEL_NAME = 'models/research/opgraph_dir/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/archisha/prototype-2.0/version2/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]

  output_dict['detection_classes'] = output_dict['detection_classes'].reshape(-1,1)
  #print(output_dict['detection_boxes'].shape)
  #print(output_dict['detection_classes'].shape)

  output_dict['detection_boxes'] = np.array(non_max_suppression_fast(remove_low_prob(merge(output_dict['detection_boxes'],output_dict['detection_classes']),output_dict['detection_scores'],0.8),0.5))
  output_dict['detection_boxes'],output_dict['detection_classes'] = output_dict['detection_boxes'][:,:-1], output_dict['detection_boxes'][:,-1].astype(int)

  #print(output_dict['detection_classes'].shape)

  output_dict['detection_scores'] = output_dict['detection_scores'][:len(output_dict['detection_boxes'])]
  output_dict['num_detections'] = len(output_dict['detection_boxes'])

  return output_dict

if __name__== '__main__':

    ###OCR ALTERNATIVE
    ocr_output = pd.DataFrame(columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num','left', 'top', 'width', 'height', 'conf', 'text'])

    cap = cv2.VideoCapture(0)
    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                ret, image_np_original = cap.read()
                #   print('Reading videoCap')
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np = np.copy(image_np_original)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                no_output = False
                try:
                    output_dict = run_inference_for_single_image(np.float32(image_np), detection_graph)
                except:
                #       print('couldnt run image inference')
                    no_output = True
                # Visualization of the results of a detection.
                if (not no_output):
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=4)
                
                cv2.imshow('Frontend Designer',cv2.resize(image_np, (800,600)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    break 
                
                    #Detection
                if(not no_output):
                    output_dict['detection_classes'] = output_dict['detection_classes'].reshape(-1,1)
                    bboxes = merge(output_dict['detection_boxes'],output_dict['detection_classes'])
                    bboxes[:,[0,1,2,3]] = bboxes[:,[1,0,3,2]]
                    html = (generate_html(bboxes,ocr_output))
                    html_file = open('output.html','w')
                    html_file.write(html)
                    # driver = webdriver.Chrome(executable_path="./chromedriver")
                    # driver.get("file:///home/archisha/prototype-2.0/version2/output.html")
                    # while True:
                    #     driver.refresh()
                    #     time.sleep(5)
                    html_file.close()
            cap.release()
                