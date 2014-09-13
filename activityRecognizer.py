import sys
import time
import pymongo
import numpy as np
import elementtree.ElementTree as ET
import activityModeler as har # this is the model generator (right now for 5 activities)

#parse xml config file
def find_in_tree(tree, node):
    found = tree.find(node)
    if found == None:
        print "No %s in file" % node
        found = []
    return found 

# Parse xml file for activityConfig 
def_file = "activityConfig.xml"
try:
    dom = ET.parse(open(def_file, "r"))
    root = dom.getroot()
except:
    print >> sys.stderr, "Unable to open and parse xml file."
    sys.exit(1)

#Parse to find the child nodes
sensorRate = find_in_tree(root,"sensorRate")
recognitionInterval = find_in_tree(root,"recognitionInterval")

#Activity Detection parameters
sensorRate = int(sensorRate.text); # in Hz
recognitionInterval = int(recognitionInterval.text) # seconds
numSamplesRecognitionInterval = sensorRate*recognitionInterval
print numSamplesRecognitionInterval 

activity_list = []
print "The activites of interest are:"
#Get list of activities from xml file
for activityName in root.findall('activity'):
	name = activityName.get('name')
	print name
	activity_list.append(name)

time.sleep(2)

# Decision Tree model
clf = har.getModel2()

MONGO_SERVER = "127.0.0.1"
MONGO_DATABASE = "mydbsensor"
MONGO_COLLECTION = sys.argv[1]
#MONGO_COLLECTION = "99000113099344"

mongodb    = pymongo.Connection(MONGO_SERVER, 27017)
database   = mongodb[MONGO_DATABASE]

'''
if MONGO_COLLECTION in database.collection_names():
  database[MONGO_COLLECTION].drop()


print "creating capped collection"
database.create_collection(
  MONGO_COLLECTION,
  size=100000,
  max=100,
  capped=True
)
'''
collection = database[MONGO_COLLECTION]


# Run this script with any parameter to add one record
# to the empty collection and see the code below
# loop correctly
#
'''
if len(sys.argv[1:]):
  collection.insert(
    {
      "key" : "value",
    }
  )
'''
# Get a tailable cursor for our looping fun
cursor = collection.find( {},
                          await_data=True,
                          tailable=True )
A = np.empty([numSamplesRecognitionInterval,3])
#B = np.empty([numSamplesRecognitionInterval,3])
count = 0;
# This will catch ctrl-c and the error thrown if
# the collection is deleted while this script is
# running.
try:

  # The cursor should remain alive, but if there
  # is nothing in the collection, it dies after the
  # first loop. Adding a single record will
  # keep the cursor alive forever as I expected.
  while cursor.alive:
    print "Top of the loop"
    try:
      for entry in cursor:
	#print entry["accel"]["x"];
      	message = cursor.next()
      	if (count == numSamplesRecognitionInterval):
		count = 0
	A[count] = ([ entry["accel"]["x"], entry["accel"]["y"], entry["accel"]["z"] ])
      	#print count
      	count = count + 1

	# extract the features for activity recognition
	features = har.getFeatureForTest(A)

	# predict the activity
	prediction = har.predictActivity(clf, features)

	# print the detected activity
	print activity_list[int(prediction-1)]
    except StopIteration:
      print "MongoDB, why you no block on read?!"
      time.sleep(1)

except pymongo.errors.OperationFailure:
  print "Delete the collection while running to see this."

except KeyboardInterrupt:
  print "trl-C Ya!"
  sys.exit(0)

print "and we're out"
