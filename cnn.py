import sys
import numpy as np

def load(caffe_root):
	sys.path.insert(0, caffe_root + 'python')
	global caffe
	import caffe
	caffe.enabled = True

def set_enabled(enable):
	caffe.enabled = enable

def disable():
	set_enabled(False)

def enable():
	set_enabled(True)

def mean(mean, binaryproto = True):
	if binaryproto:
		from caffe.proto import caffe_pb2
		from caffe.io import blobproto_to_array

		# Transform a protoblob to a numpy array
		blob = caffe_pb2.BlobProto()
		data = open(mean, "rb").read()
		blob.ParseFromString(data)
		nparray = blobproto_to_array(blob)
	else:
		np.load(mean)

	return nparray

def init(model, pretrained, npmean, **kwargs):
	if not caffe.enabled:
		return Classifier()

	return caffe.Classifier(model, pretrained,
		mean=npmean,
		**kwargs)

class Classifier():
	def predict(self, ims):
		return -1
