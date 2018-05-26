import csv

fileDir = "C:\\Users\\Nitsan\\Desktop\\School\\thesis\\statistics"
path = fileDir+"\dataMatrix.csv"

csvFile = open(path, 'r')

allFile = csv.reader(csvFile)

lineCounter = 0
missingIndices = []
for line in allFile:
	#if lineCounter > 50:
	#	break
	#relevant only for question regarding classifications
	if lineCounter == 0:
		countH = 0
		countM = 0
		countNull = 0
		for classfier in line:
			if classfier == 'H':
				countH += 1
			if classfier == 'M':
				countM += 1
		print ("amount M: " + str(countM))
		print ("amount H: " + str(countH))
		sumOfRanksH = 0
		sumOfRanksM = 0
		for index in range(0,len(line)):
			if line[index] == 'H':
				sumOfRanksH += index
			if line[index] == 'M':
				sumOfRanksM += index
		print("Sum of ranks H: " + str(sumOfRanksH))
		print("Sum of ranks M: " + str(sumOfRanksM))
			
	lineDetails = list(set(line))
	
	#Collect missing indices

	if lineCounter > 0:
		#print(line)
		#print("-----------------")
		#print(len(lineDetails))
		#print("-----------------")
		#print(lineDetails)
		for x in range(0,len(line)):
			if line[x] == '':
				obj = type('obj', (object,), {'probe' : line[0], 'lineNumber': lineCounter})
				missingIndices.append(obj)
				
	lineCounter += 1
	
missingIndices = list(set(missingIndices))
print(len(missingIndices))
print ("Missing Indices: ")
missingIndices.sort(key=lambda x: x.lineNumber, reverse=False)
for missing in missingIndices:
	print("Probe: " + missing.probe + " line number " + str(missing.lineNumber+1))
	
print("amount missing indices: " + str(len(missingIndices)))



