import csv


def printDebug(mode,val):
	if mode == 'debug':
		print('DEBUG: ' + str(val))
	elif mode == 'info':
		print('INFO: ' + str(val))


fileDir = "/Users/nitsan/programming/Thesis/"
path = fileDir+"dataMatrix_afterReduction.csv"
outputFileDir = "/Users/nitsan/programming/Thesis/statisticsAssignment/rs_g/"
outputFileName = 'rs_g1.csv'

csvFile = open(path, 'rw')

allFile = csv.reader(csvFile)

outputFile = open(outputFileDir+outputFileName, 'wr')
outputFile.close()

outputFile = open(outputFileDir+outputFileName, 'a')

printMode = 'info'

lineCounter = 0
missingIndices = []
indexArray = []
for line in allFile:
	if lineCounter == 0:
		for word in line:
			indexArray.append(word)
	elif lineCounter > 1:
		indexCounter = 0
		mCounter = 0
		hCounter = 0
		for score in line:
			if printMode == 'debug':
				print score			
			if printMode == 'debug':
				print ('indexCounter: '+ str(indexCounter) + ' indexArray: '+str(len(indexArray)))
			if indexCounter > 1:
				if indexArray[indexCounter] == 'M':
					mCounter += float(score)
				else:
					hCounter += float(score)
			indexCounter += 1
	if lineCounter > 0:
		printDebug('debug',str(line[1:]).replace('[','').replace(']','').replace("'",''))
		outputFile.write(str(line[1:]).replace('[','').replace(']','').replace("'",''))
		outputFile.write('\n')		
	lineCounter += 1

outputFile.close()		
if printMode == 'debug':
	print(indexArray)
			





