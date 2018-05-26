import csv

fileDir = "/Users/nitsan/programming/Thesis/"
path = fileDir+"dataMatrix_afterReduction.csv"
outputFileDir = "/Users/nitsan/programming/Thesis/statisticsAssignment/rs_g/"
outputFileName = 'rs_g.csv'

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
		if printMode == 'debug':
			print(str(mCounter) + '------' + str(hCounter))
		if printMode == 'debug' or printMode == 'info':
			print lineCounter
		outputFile.write(str(mCounter))
		outputFile.write(',')
		outputFile.write(str(hCounter))
		outputFile.write(',')
		outputFile.write(str(mCounter - hCounter))
		outputFile.write(',')
		outputFile.write(str(abs(mCounter - hCounter)))
		outputFile.write(',')
		if (mCounter - hCounter) > 0:
			outputFile.write('1')
		else:
			outputFile.write('0')
		outputFile.write('\n')			
			
	lineCounter += 1

outputFile.close()		
if printMode == 'debug':
	print(indexArray)
			
		



