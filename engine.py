import os
import json
import urllib

import h5py
import numpy as np
import pickle as pk

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import get_file

# Load models and support

damage_gate = load_model('data/ft_model_1.h5')
print("Damage gate loaded")

severity_model = load_model('data/ft_model_3.h5')
print("Severity model loaded")


def prepare_img_256(img_path):
	urllib.request.urlretrieve(img_path, 'save.png') # or other way to upload image
	img = load_img('save.png', target_size=(256, 256)) # this is a PIL image 
	x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
	x = x.reshape((1,) + x.shape)/255
	return x

def car_damage_gate(img_256, model):
	print("Validating that damage exists...")
	pred = model.predict(img_256)
	if pred[0][0] <=.5:
		return True # print "Validation complete - proceed to location and severity determination"
	else:
		return False
		# print "Are you sure that your car is damaged? Please submit another picture of the damage."
		# print "Hint: Try zooming in/out, using a different angle or different lighting"

def severity_assessment(img_256, model):
	print("Determining severity of damage...")
	pred = model.predict(img_256)
	pred_label = np.argmax(pred, axis=1)
	d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
	for key in d.keys():
		if pred_label[0] == key:
			return d[key]
	# 		print "Assessment: {} damage to vehicle".format(d[key])
	# print "Severity assessment complete."

# load models
def engine(img_path):
	img_256 = prepare_img_256(img_path)
	g2 = car_damage_gate(img_256, damage_gate)

	if g2 is False:
		result = {
		'gate2': 'Damage presence check: ',
		'gate2_result': 0,
		'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
		1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
		return result
	
	y = severity_assessment(img_256, severity_model)
	
	result = {
	'gate2': 'Damage presence check: ',
	'gate2_result': 1,
	'gate2_message': {0: None, 1: None},
	'severity': y,
	'final': 'Damage assessment complete!'}
	return result