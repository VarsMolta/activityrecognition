import numpy as np
import scipy
#from sklearn import svm, cross_validation, tree
from sklearn import tree
import csv
import StringIO
import pylab as pl # for plotting
from sklearn.lda import LDA

'''
takes in a filename
returns a 2D list with the contents of the file
'''
def readCSV(filename):
	with open(filename,"rU") as csvfile:
		rows = []
		reader = csv.reader(csvfile, delimiter=';')
		for line in reader:
			rows.append(line)
	print "%d rows in %s" % (len(rows), filename)
	return rows

# mean absolute deviation
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

# zero crossing rate
def zcr(data, numElements, numCols, axis=None):
	(z,i) = np.where(np.diff(np.sign(data), 1, axis))
	temp = np.zeros(numCols)
	for iteration in range(0,numCols):
		zero_crossings = z[i==iteration]
		zcRate = float(np.size(zero_crossings))/numElements
		temp[iteration] = zcRate
	
	return temp

# mean crossing rate
def mcr(data, numElements, numCols, axis=None):
    normData = (data - np.mean(data, axis))
    return zcr(normData, numElements, numCols, axis)

'''
takes in data for different activities
returns a 2D array of feature vectors for each of the activities
'''
def getFeatures(walkData, drinkData, typeData, eatData, tvData, startSeconds, stopSeconds, freqHz, feature, iteration):
		print startSeconds 
		print stopSeconds
		wdata = walkData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		ddata = drinkData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		tdata = typeData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		edata = eatData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		tvdata = tvData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
	
		alldata = np.concatenate((wdata, ddata, tdata, edata, tvdata), axis=1)
		print alldata
		#rows = np.array(data[:,1]).astype(np.float)
		
		npmean = np.mean(alldata, axis=0)  # feature 1
		feature = npmean

		nprms = np.sqrt(np.mean(alldata**2, axis=0)) # feature 2
		feature = np.vstack([feature,nprms])
		#print np.sqrt(np.mean(alldata**2, axis=0))
		
		npstd = np.std(alldata, axis=0) # feature 3
		feature = np.vstack([feature,npstd])
		#print np.std(alldata, axis=0)

		npvar = np.var(alldata, axis=0) # feature 4
		feature = np.vstack([feature,npvar])
		#print np.var(alldata, axis=0)
		
		npmad = mad(alldata, axis = 0) # feature 5
		feature = np.vstack([feature,npmad])
		
		[numRows, numCols] = np.shape(alldata)
		npzcr = zcr(alldata, numRows, numCols, axis = 0) # feature 6
		feature = np.vstack([feature,npzcr])
		
		npmcr = mcr(alldata, numRows, numCols, axis = 0) # feature 7
		feature = np.vstack([feature,npmcr])
		return feature

def getModel(feature, numFeatures, numSamples):
	
		fC1 = []
		fC2 = []
		fC3 = []	
		fC4 = []	
		fC5 = []	
		for iteration in range(0,numSamples):
			fc1 = feature[iteration*numFeatures:(iteration+1)*numFeatures,0:3:].flatten()
			fc2 = feature[iteration*numFeatures:(iteration+1)*numFeatures,3:6:].flatten()
			fc3 = feature[iteration*numFeatures:(iteration+1)*numFeatures,6:9:].flatten()
			fc4 = feature[iteration*numFeatures:(iteration+1)*numFeatures,9:12:].flatten()
			fc5 = feature[iteration*numFeatures:(iteration+1)*numFeatures,12:15:].flatten()
			# don't append if first time thru the loop
			if iteration == 0:
				fC1 = fc1;
				fC2 = fc2;
				fC3 = fc3;
				fC4 = fc4;
				fC5 = fc5;
			else:
				fC1 = np.vstack([fC1,fc1])
				fC2 = np.vstack([fC2,fc2])
				fC3 = np.vstack([fC3,fc3])
				fC4 = np.vstack([fC4,fc4])
				fC5 = np.vstack([fC5,fc5])
		#print mad(alldata, axis = 0)
		#print "YYYYY"
		#print fC1 
	
		X = np.vstack([fC1,fC2]);	
		X = np.vstack([X,fC3]);
		X = np.vstack([X,fC4]);
		X = np.vstack([X,fC5]);
		
		print "Samples vs Feature Matrix"
		print X;	
		# insert for loop here
		#hist, bin_edges = np.histogram(alldata[:,0],bins=10)
		#print np.cumsum(hist)

		#print np.percentile(alldata, 50, axis = 0) #IGNORE
		#numRows, numCols = np.shape(alldata)
		#print mcr(alldata[:,0], numRows, axis=0)
		Y = np.array([1*np.ones(numSamples),2*np.ones(numSamples), 3*np.ones(numSamples), 4*np.ones(numSamples), 5*np.ones(numSamples)])
		Y = Y.flatten()	

		target_names = np.array(["walking", "drinking", "typing", "eating", "remoteTV"])
		lda = LDA(n_components=2)
		X_r2 = lda.fit(X, Y).transform(X)

		''' Do LDA analysis on this feature set.
		Code from here: http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
		'''
		print "Printing LDA params"
		print np.shape(X_r2);
		print (X_r2)
		
		pl.figure()
		for c, i, target_name in zip("rgbyk", [1, 2, 3, 4, 5], target_names):
    			pl.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], c=c, label=target_name)
		
		pl.legend()
		pl.title('LDA of Human Activity dataset')

		pl.show()
		
		#clf = tree.DecisionTreeClassifier(random_state=22)
		clf = tree.DecisionTreeClassifier()
		print clf.get_params()['random_state']
		clf = clf.fit(X, Y)
	
		return clf


def getFeatureForTest(walkData, drinkData, typeData, eatData, tvData, startSeconds, stopSeconds, freqHz):


		wdata = walkData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		ddata = drinkData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		tdata = typeData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		edata = eatData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
		tvdata = tvData[startSeconds:stopSeconds:freqHz,1:4:].astype(float)
	
		alldata = np.concatenate((wdata, ddata, tdata, edata, tvdata), axis=1)
		#print alldata
		#rows = np.array(data[:,1]).astype(np.float)
		
		npmean = np.mean(alldata, axis=0)
		feature = npmean 

		nprms = np.sqrt(np.mean(alldata**2, axis=0))
		feature = np.vstack([feature,nprms])
		#print np.sqrt(np.mean(alldata**2, axis=0))
		
		npstd = np.std(alldata, axis=0)
		feature = np.vstack([feature,npstd])
		#print np.std(alldata, axis=0)

		npvar = np.var(alldata, axis=0)
		feature = np.vstack([feature,npvar])
		#print np.var(alldata, axis=0)
		
		npmad = mad(alldata, axis = 0)
		feature = np.vstack([feature,npmad])
		
		[numRows, numCols] = np.shape(alldata)
		npzcr = zcr(alldata, numRows, numCols, axis = 0) # feature 6
		feature = np.vstack([feature,npzcr])
		
		npmcr = mcr(alldata, numRows, numCols, axis = 0) # feature 7
		feature = np.vstack([feature,npmcr])

		fc1 = feature[:,0:3:].flatten()
		fc2 = feature[:,3:6:].flatten()
		fc3 = feature[:,6:9:].flatten()
		fc4 = feature[:,9:12:].flatten()
		fc5 = feature[:,12:15:].flatten()

		#uncomment one of the following statements to test (fc1 for walking, fc2 for drinking, fc3 for typing)
		#return fc1
		#return fc2
		#return fc3
		#return fc4
		return fc5

def main():
	# read input csv files for walking, drinking and typing trained data
	walk_filename = "walking.csv"
	walk_rows = readCSV(walk_filename)
	walkData = np.array(walk_rows)
	
	drink_filename = "drinking.csv"
	drink_rows = readCSV(drink_filename)
	drinkData = np.array(drink_rows)
	
	type_filename = "typing.csv"
	type_rows = readCSV(type_filename)
	typeData = np.array(type_rows)

	eat_filename = "eating2.csv"
	eat_rows = readCSV(eat_filename)
	eatData = np.array(eat_rows)
	
	tv_filename = "remoteTV.csv"
	tv_rows = readCSV(tv_filename)
	tvData = np.array(tv_rows)
	
	#initialize empty list
	feature2 = []
	feature = []
	
	startSeconds = 0;
	endSeconds = 15;
	overlap = 2;
	freqDecimate100 = 2;
	numSecondsPerCalc = 4;
	numSamples = 48;
	numFeatures = 7;

	for iteration in range(0,numSamples): 	
		feature1 = getFeatures(walkData, drinkData, typeData, eatData, tvData, 100*(startSeconds+overlap*iteration), 100*(startSeconds+numSecondsPerCalc+overlap*iteration), freqDecimate100, feature2,0)
			
		# don't append if first time thru the loop
		if iteration == 0:
			feature = feature1;
		else:
			feature = np.vstack([feature,feature1])

	#print "XXXXXX"
	#print feature

	clf = getModel(feature, numFeatures, numSamples)

	print "Prediction ([1] = Walking, [2] = Drinking, [3] = Typing, [4] = Eating, [5] = Watching TV with Remote in Hand). Change return value output of getFeaturesForTest for testing all activities"
	# this will test the model from 17 seconds to 27 seconds in 4s intervals with 2s overlap
	for x in range (0,7):
		fc2 = getFeatureForTest(walkData, drinkData, typeData, eatData, tvData, 8000+x*200, 8000+x*200+400, freqDecimate100 )
		prediction = clf.predict(fc2)
		print prediction


if __name__=='__main__':
	main()

