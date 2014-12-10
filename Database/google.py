import json
import os
import time
import requests
import string
import random
import traceback
import sys
from PIL import Image
from StringIO import StringIO
from requests.exceptions import ConnectionError

try:
	import gevent
	from gevent import monkey
	using_gevent = True
except:
	using_gevent = False
	print "[FAIL] Could not load gevent, no Timeout will be used."
	print "\tInstalling (conda install gevent | pip install gevent) is adviced."


def get(url):
	def _(url):
		r = None
		try:
			r = requests.get(url)
		except ConnectionError, e:
			print '\t[FAIL] Could not download %s' % url
		return r

	r = None
	if using_gevent:
		with gevent.Timeout(5, False):
			r = _(url)
	else:
		r = _(url)

	return r


def validate(filename):
	filename.replace(" ", "_")
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	return ''.join(c for c in filename if c in valid_chars)[:12]
 
def go(query, folder):
	"""Download full size images from Google image search.
 
	Don't print or republish images without permission.
	I used this to train a learning algorithm.
	"""
	BASE_URL = 'https://ajax.googleapis.com/ajax/services/search/images?'\
						 'v=1.0&q=' + query + '&start=%d'
 
 
	BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder)
 
	if not os.path.exists(BASE_PATH):
		os.makedirs(BASE_PATH)
 
	start = 0 # Google's start query string parameter for pagination.
	while start < 60: # Google will only return a max of 56 results.
		try:
			r = get(BASE_URL % start)
			if r == None:
				print "\t[FAIL] Request timed out: %s" % (BASE_URL % start)
				continue

			js = json.loads(r.text)

			if js is None or not 'responseData' in js or js['responseData'] is None:
				print '\t[FAIL] JSON response is invalid.'
				print '\t[DETAIL] ', js['responseDetails']
				exit(0)

			for image_info in js['responseData']['results']:

				sys.stdout.write("\t[%%] %.2f\r" % (start*100.0/64.0,))
				sys.stdout.flush()
				start = start + 1

				url = image_info['unescapedUrl']

				image_r = get(url)
				if image_r == None:
					print "\t[FAIL] Request timed out: %s" % (url)
					continue
	 
				# Remove file-system path characters from name.
				title = image_info['imageId']
				validate(title)
				file = open(os.path.join(BASE_PATH, '%s.jpg') % title, 'wb+')
				try:
					Image.open(StringIO(image_r.content)).save(file, 'JPEG')
				except IOError, e:
					# Throw away some gifs...blegh.
					print '\t[FAIL] Could not save %s' % title
					continue
				finally:
					file.close()

		except Exception as exc:
			print '\t[FAIL] ', exc
			exc_type, exc_value, exc_traceback = sys.exc_info()
			traceback.print_tb(exc_traceback)
			continue

		# Be nice to Google and they'll be nice back :)
		time.sleep(1.5)

	sys.stdout.write("\t[%%] %.2f\n" % (100.0,))

# Patch eventlet
if using_gevent:
	try:
		gevent.monkey.patch_all()
		print "[OK] Using Timeouts"
	except Exception as exc:
		print "[FAIL] Could not call gevent.monkey.patch_all()."
		print "\tTry reinstalling (conda install gevent | pip install gevent)."
		print exc

def start(folder, slug):
	# Done
	with open("done" + slug + ".txt", "a+") as d:
		d.seek(0, 0)
		done = [l.strip() for l in d]

	# Example use
	with open('words' + slug + '.txt', 'r') as f:
		unique = set([l.strip() for s in f for l in s.split("\t") if l not in done])
		dic = [w for w in unique]
		
	random.shuffle(dic)

	with open("done" + slug + ".txt", "a") as d:
		for w in dic:
			print "\n[WORD] %s" % w

			# Save to done list
			d.write(w + "\n")

			# Get images
			go(w, folder)

	print "[OK] All images have been downloaded"


# Autoload
autoload = raw_input("Autoload file? [FILE / blank]? ")
if autoload == "":

	# Ask for folder
	folder = raw_input("Where do you want to save to? ")

	# Ask for file slug
	slug = raw_input("Which file slug would you want to use (word + slug + .txt)? ")

else:
	with open(autoload, 'r') as f:
		for l in f:
			l = l.strip()
			start('images/' + l, l)
