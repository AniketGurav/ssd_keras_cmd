from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import pickle
from scipy.misc import imread
from ssd import SSD300
from ssd_utils import BBoxUtility
from optparse import OptionParser
import os
import cv2

np.set_printoptions(suppress=True)

NUM_CLASSES = 3
acsl_classes = ['person', 'car']
colors = [[255, 0, 0], [0, 255, 0]]

input_shape = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

parser = OptionParser()

parser.add_option("--model_path", dest="model_path", help="Path to test model.")
parser.add_option("--input_path", dest="input_path", help="Input path to test data.")
parser.add_option("--output_path", dest="output_path", help="Output path to test data.")

options, _ = parser.parse_args()

model_path = options.model_path
input_path = options.input_path
output_path = options.output_path

inputs = []
images = []

img_files = [f for f in os.listdir(input_path)]

for img_file in img_files:
    img_path = os.path.join(input_path, img_file)
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())

inputs = preprocess_input(np.array(inputs))

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(model_path, by_name=True)

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

for i, img in enumerate(images):
    # Parse the outputs.
    if isinstance(results[i], list) and not results[i]:
        continue
	
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [k for k, conf in enumerate(det_conf) if conf >= 0.3]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    for j in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[j] * img.shape[1]))
        ymin = int(round(top_ymin[j] * img.shape[0]))
        xmax = int(round(top_xmax[j] * img.shape[1]))
        ymax = int(round(top_ymax[j] * img.shape[0]))
        score = top_conf[j]
        label = int(top_label_indices[j])
        label_name = acsl_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label - 1]

	cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (color[0], color[1], color[2]), 2)
	cv2.putText(img, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.3, (color[0], color[1], color[2]), 1)

    cv2.imwrite(os.path.join(output_path, '{}.jpg'.format(i)), img)
