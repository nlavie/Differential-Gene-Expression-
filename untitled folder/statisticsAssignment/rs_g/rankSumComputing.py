import csv


def printDebug(mode,val):
	if mode == 'debug':
		print('DEBUG: ' + str(val))
	elif mode == 'info':
		print('INFO: ' + str(val))

def calculateMinus(listA,listB):
	listC = []
	for i in range(1,len(listB)-1):
		print(i)
		if(listA[i] != '.' and listB[i] != '.' and listA[i] != ',' and listB[i] != ',' and listA[i] != ' ' and listB[i] != ' '):
			listC.append(float(listA[i]) - float(listB[i]))
		else:
			listC.append('')
	return listC

def replaceInvalidChars(val):
	return val.replace("'",'').replace('[','').replace(']','')


fileDir = "/Users/nitsan/programming/Thesis/statisticsAssignment/rs_g/"
path = fileDir+"rs_g1.csv"
outputFileDir = "/Users/nitsan/programming/Thesis/statisticsAssignment/rs_g/"
outputFileName = 'rs_g2.csv'

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
	elif lineCounter >1:
		for testLine in allFile:
			printDebug('debug','line' + str(line))
			printDebug('debug','testLine' + str(testLine))
			printDebug('debug',replaceInvalidChars(str(calculateMinus(replaceInvalidChars(str(line)),replaceInvalidChars(str(testLine))))))
			outputFile.write(replaceInvalidChars(str(calculateMinus(replaceInvalidChars(str(line)),replaceInvalidChars(str(testLine))))))
			outputFile.write('\n')

	lineCounter += 1

outputFile.close()		
if printMode == 'debug':
	print(indexArray)
			





