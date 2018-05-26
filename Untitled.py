
# coding: utf-8

# In[1]:


import pandas as pd
from IPython.display import display
import csv
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import scipy as scp
from scipy import stats as stts


# In[7]:


df = pd.read_csv('~/nitsan/untitled folder/statisticsAssignment/rs_g/rs_g1.csv')


# In[21]:


outPutFile = open(('~/nitsan/untitled folder/statisticsAssignment/rs_g/rs_g5.csv','wr'))


# In[23]:


dfFull = pd.read_csv('~/nitsan/untitled folder/statisticsAssignment/rs_g/dataMatrix_forDE.csv')


# In[24]:


subtractDf = pd.DataFrame


# In[25]:


dfFull


# In[384]:


print (dfFull.loc[0][1])


# In[92]:


# Calculating P value and t test -WRS


# In[291]:


# Calculating P value and t test -WRS
#WRS
listHVal = []
listMVal = []
listHOver = []
listHUnder = []
listMOver = []
listMUnder = []
for i in range(0,54627):
    tmpHList = []
    tmpMList = []
    for j in range(1,100):     
        if j < 51:
            tmpHList.append(dfFull.loc[i][j])
        else:
            tmpMList.append(dfFull.loc[j][j])
    m2h = stts.ranksums(tmpMList,tmpHList)
    m2hP = m2h.pvalue
    m2hS = m2h.statistic
    listMVal.append(m2hP)
    h2m =  stts.ranksums(tmpHList,tmpMList)
    h2mP = h2m.pvalue
    h2mS = h2m.statistic
    listHVal.append(h2mP)
    if m2hS < m2hP:
        listMOver.append(m2hP)
    else:
        listMUnder.append(m2hP)
    if h2mS < h2mP:
        listHOver.append(h2mP)
    else:
        listHUnder.append(h2mP)
    
    if i % 500 == 0 or i == 54627:
        print('Counter: ' + str(i) + ' out of 54627' + ' Percentage ' + str(i/54627)) 


# In[292]:


fig = plt.figure()
plt.xlabel('WRS P-Value for M vs H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMVal)


# In[130]:


listMValForE = listMVal
listMValForE= listMValForE.sort()
listMValForESummed = []
for c in range(1,len(listMValForE)):
    if c == 0:
        listMValForESummed.append(listMValForE[c])
    else:
        listMValForESummed.append(2*c-1)


# In[196]:





# In[280]:


vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []


# In[295]:


listMValForE = []
listMValForE.extend(listMVal)
listMValForE.sort()
listMValForESummed = []
listNormalLine = []
for c in range(0,len(listMValForE)):
    if c == 0:
        listMValForESummed.append(1)
        listNormalLine.append(c)
    else:
        listMValForESummed.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listMValForE:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(5000)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+5000)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('WRS P-Value for for M vs H')
plt.ylabel('Count')


# In[285]:


listMValOverForE = []
listMValOverForE.extend(listMOver)
listMValOverForE.sort()
listMValOverForESummed = []
listNormalLine = []
for c in range(0,len(listMValOverForE)):
    if c == 0:
        listMValOverForESummed.append(1)
        listNormalLine.append(c)
    else:
        listMValOverForESummed.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listMValOverForE:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(3600)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+3600)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('WRS P-Value for Overexprssed in M')
plt.ylabel('Count')


# In[290]:


listHValOverForE = []
listHValOverForE.extend(listHOver)
listHValOverForE.sort()
listHValOverForESummed = []
listNormalLine = []
for c in range(0,len(listHValOverForE)):
    if c == 0:
        listHValOverForESummed.append(1)
        listNormalLine.append(c)
    else:
        listHValOverForESummed.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listHValOverForE:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(2000)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+2000)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('WRS P-Value for Overexprssed in H')
plt.ylabel('Count')


# In[ ]:





# In[ ]:


listHValOverForE = listHValOver
listHValOverForE= listHValOverForE.sort()
listHValOverForESummed = []
for c in range(1,len(listHValOverForE)):
    if c == 0:
        listMValForESummed.append(listHValOverForE[c])
    else:
        listMValForESummed.append(2*c-1)


# In[93]:


tmpHList1 = []
tmpMList1 = []
for i in range(0,1):
    for j in range(1,100):     
        if j < 51:
            tmpHList1.append(dfFull.loc[i][j])
        else:
            tmpMList1.append(dfFull.loc[j][j])
true_mu = 0

stts.ttest_ind(tmpMList1,tmpHList1)


# In[296]:


# Calculating P value and t test -WRS
#TTest
listHVal1 = []
listMVal1 = []
listHOver1 = []
listHUnder1 = []
listMOver1 = []
listMUnder1 = []
for i in range(0,54627):
    tmpHList = []
    tmpMList = []
    for j in range(1,100):     
        if j < 51:
            tmpHList.append(dfFull.loc[i][j])
        else:
            tmpMList.append(dfFull.loc[j][j])
    m2h = stts.ttest_ind(tmpMList,tmpHList)
    m2hP = m2h.pvalue
    m2hS = m2h.statistic
    listMVal1.append(m2hP)
    h2m =  stts.ttest_ind(tmpHList,tmpMList)
    h2mP = h2m.pvalue
    h2mS = h2m.statistic
    listHVal1.append(h2mP)
    if m2hS < m2hP:
        listMOver1.append(m2hP)
    else:
        listMUnder1.append(m2hP)
    if h2mS < h2mP:
        listHOver1.append(h2mP)
    else:
        listHUnder1.append(h2mP)
    if i % 1000 == 0:
        print('Counter: ' + str(i) + ' out of 54627' + ' Percentage ' + str(i/54627))
print('Counter: ' + str(i) + ' out of 54627' + ' Percentage ' + str(i/54627)) 


# In[300]:


listMValForE12 = []
for y in range(1,len(listMValForE1)):
    if y % 2 == 0:
        listMValForE12.append(listMValForE1[y])
print('done')


# In[302]:


print(len(listMValForE12))


# In[304]:


listMValForE1 = []
listMValForE1.extend(listMValForE12)
listMValForE1.sort()
listMValForESummed1 = []
listNormalLine = []
for c in range(0,len(listMValForE1)):
    if c == 0:
        listMValForESummed1.append(1)
        listNormalLine.append(c)
    else:
        listMValForESummed1.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listMValForE1:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(5500)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+5500)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('Ttest P-Value for for M vs H')
plt.ylabel('Count')


# In[308]:


listMValOverForE1 = []
listMValOverForE1.extend(listMOver1)
listMValOverForE1.sort()
listMValOverForESummed1 = []
listNormalLine = []
for c in range(0,len(listMValOverForE1)):
    if c == 0:
        listMValOverForESummed1.append(1)
        listNormalLine.append(c)
    else:
        listMValOverForESummed1.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listMValOverForE1:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(3400)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+3400)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('Ttest P-Value for Overexpressed in M')
plt.ylabel('Count')


# In[450]:


listMValUnderForE1 = []
listMValUnderForE1.extend(listHOver1)
listMValUnderForE1.sort()
listMValUnderForESummed1 = []
listNormalLine = []
for c in range(0,len(listMValUnderForE1)):
    if c == 0:
        listMValUnderForESummed1.append(1)
        listNormalLine.append(c)
    else:
        listMValUnderForESummed1.append(c+1)
        listNormalLine.append(c)

    
#listMValOverForE.append(1)
#listNormalLine.append(1)
print('done')
vals = np.arange(0.0, 1.0, 0.1)
valsCounters = [0]*10
valsSummed = []
sum = []
normalLine = []
for val in listMValUnderForE1:
    for rngIndex in range(0,len(vals)-1):
        if val >= vals[rngIndex] and val <= vals[rngIndex+1]:
            valsCounters[rngIndex] = valsCounters[rngIndex]+1
print(valsCounters)
for valIndex in range(0,len(valsCounters)):
    if valIndex == 0:
        sum.append(valsCounters[0])
        normalLine.append(2280)
    else:
        sum.append(valsCounters[valIndex] + sum[valIndex-1])
        normalLine.append(normalLine[valIndex-1]+2280)
print(sum)
print(normalLine)
print('done')
plt.plot(vals,sum, c='green')
plt.plot(vals,normalLine, c='blue')
plt.xlabel('Ttest P-Value for Overexpressed in H')
plt.ylabel('Count')


# In[34]:


figMOverTtest = plt.figure()
plt.xlabel('Ttest P-Value for Overexprssed in M')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMOver1)


# In[35]:


figMUnderTtest = plt.figure()
plt.xlabel('Ttest P-Value for Underexpressed in M')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMUnder1)


# In[37]:


figHOverTtest = plt.figure()
plt.xlabel('Ttest P-Value for Overexprssed in H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listHOver1)


# In[38]:


figHUnderTtest = plt.figure()
plt.xlabel('Ttest P-Value for Underexpressed in H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listHUnder1)


# In[28]:


figMOver = plt.figure()
plt.xlabel('WRS P-Value for Overexprssed in M')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMOver)


# In[29]:


figMUnder = plt.figure()
plt.xlabel('WRS P-Value for Underexpressed in M')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMUnder)


# In[30]:


figHOver = plt.figure()
plt.xlabel('WRS P-Value for Overexprssed in H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listHOver)


# In[31]:


figHUnder = plt.figure()
plt.xlabel('WRS P-Value for Underexpressd in H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listHUnder)


# In[96]:


fig1 = plt.figure()
plt.xlabel('T test P-Value for M vs H')
plt.ylabel('Count')
n, bins, patches = plt.hist(listMVal1)


# In[34]:





# In[ ]:


subtractDf.loc[1]


# In[ ]:


for i in range(1,49):
    print(stts.ranksums(df['H'],df['M.'+str(i)]))
for j in range(1,50):
    print(stts.ranksums(df['M'],df['H.'+str(j)]))
for k in range(1,49):
    for m in range(1,49):
        print(stts.ranksums(df['H.'+str(k)],df['M.'+str(m)]))


# In[358]:


# Calculating P value and t test -WRS
#WRS
listHVal2 = []
listMVal2 = []
listHOver2 = []
listHUnder2 = []
listMOver2 = []
listMUnder2 = []
listC = []
listForConclusion = []
overCounter = 0
underCounter = 0
for i in range(0,54627):
    if len(listMOver2) == 60 and len(listMUnder2) == 60:
        break
        
    tmpHList = []
    tmpMList = []
    for j in range(1,100):     
        if j < 51:
            tmpHList.append(dfFull.loc[i][j])
        else:
            tmpMList.append(dfFull.loc[j][j])
    m2h = stts.ranksums(tmpMList,tmpHList)
    m2hP = m2h.pvalue
    m2hS = m2h.statistic
    h2m =  stts.ranksums(tmpHList,tmpMList)
    h2mP = h2m.pvalue
    h2mS = h2m.statistic
    tmpCList = []
    print('Over counter: ' + str(overCounter) + ' Under counter: ' + str(underCounter))
    added = False
    if m2hS < m2hP and overCounter < 60:
        listMOver2.append(m2hP)
        tmpCList.extend(tmpMList)
        tmpCList.extend(tmpHList)
        overCounter = overCounter + 1
        if len(listForConclusion) < 2:
            listForConclusion.append(tmpCList)
        added=True
    elif underCounter < 60 and overCounter == 60:
        listMUnder2.append(m2hP)
        tmpCList.extend(tmpMList)
        tmpCList.extend(tmpHList)
        underCounter = underCounter + 1
        if len(listForConclusion) < 3:
            listForConclusion.append(tmpCList)
        added=True
    if h2mS < h2mP:
        listHOver2.append(h2mP)
    else:
        listHUnder2.append(h2mP)
    if added == True:
        listC.append(tmpCList)
    
    if i % 500 == 0 or i == 54627:
        print('Counter: ' + str(i) + ' out of 54627' + ' Percentage ' + str(i/54627))
print('Over counter: ' + str(len(listMOver2)) + ' Under counter: ' + str(len(listMUnder2)) + ' Total ' + str(len(listC)))


# In[359]:


len(listForConclusion)


# In[105]:


len(listMOver2)


# In[109]:


corMatrix = [None]*120
for i in range(0,len(listC)):
    oneGeneList = [None]*120
    for j in range(0,len(listC)):
        if i != j:
            oneGeneList[j]=stts.spearmanr(listC[i],listC[j]).correlation
        else:
            oneGeneList[j]=0
    corMatrix[i]=oneGeneList
    if i % 10 == 0:
        print ('Counter: ' + str(i) + ' Out of ' + str(len(listC)))


# In[65]:


corMatrix[0]


# In[66]:


corrletionCsv = open('corrletionCsv.csv', 'w')


# In[84]:


import seaborn as sns


# In[97]:


sns.set(style= 'white')
#mask = np.zeros_like(corMatrix, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corMatrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5})


# In[95]:


sns.heatmap(corMatrix, 
            xticklabels=corMatrix.columns.values,
            yticklabels=corMatrix.columns.values)


# In[379]:


plt.matshow(corMatrix)
plt.colorbar()


# In[447]:


ax3 = sns.heatmap(corMatrix)


# In[77]:


for i in range(1,len(corMatrix)):
    corrletionCsv.write(str(corMatrix[i]).replace('[','').replace(']',''))
    corrletionCsv.write('\n')


# In[128]:


corMatrixH = [None]*120
for i in range(0,len(listC)):
    oneGeneList = [None]*120
    for j in range(0,len(listC)):
        if i != j and j < 51:
            oneGeneList[j]=stts.spearmanr(listC[i],listC[j]).correlation
        else:
            oneGeneList[j]=0
    corMatrixH[i]=oneGeneList
    if i % 10 == 0:
        print ('Counter: ' + str(i) + ' Out of ' + str(len(listC)))


# In[122]:


corMatrix[50][49]


# In[378]:


plt.matshow(corMatrixH)
plt.colorbar()


# In[448]:


ax4 = sns.heatmap(corMatrixH)


# In[344]:


def findFdr(listVals):
    listVals = sorted(listVals)
    maxIndex1 = 0
    maxIndex2 = 0
    maxIndex3 = 0
    a1 = 0.1
    a2 = 0.05
    a3 = 0.001
    for index in range(0,len(listVals)):
        if listVals[index] <= a1*index/len(listVals):
            maxIndex1 = index
        if listVals[index] <= a2*index/len(listVals):
            maxIndex2 = index
        if listVals[index] <= a3*index/len(listVals):
            maxIndex3 = index
    print('FDR: '+ str(a1) + ' maxIndex: ' +str(maxIndex1))# + ' result: ' + str(len(listVals)-maxIndex1))
    print('FDR: '+ str(a2) + ' maxIndex: ' +str(maxIndex2))#+ ' result: ' + str(len(listVals)-maxIndex2))
    print('FDR: '+ str(a3) + ' maxIndex: ' +str(maxIndex3))#+ ' result: ' + str(len(listVals)-maxIndex3))
        


# In[332]:


t = [3,2,1]
sorted(t)
t


# In[346]:



print('WRS - M General')
findFdr(listMVal)
print('WRS - Overexpressed in M')
findFdr(listMOver)
print('WRS - Underexpressed in M')
findFdr(listMUnder)
print('Ttest - M General')
findFdr(listMVal1)
print('Ttest - Overexpressed in M')
findFdr(listMOver1)
print('Ttest - Underexpressed in M')
findFdr(listMUnder1)


# In[ ]:


listForConclusion


# In[364]:


smallCorMatrix = []
for d in range (0,len(listForConclusion)):
    corLine = []
    for d1 in range (1,len(listC)):
        corLine.append(stts.spearmanr(listForConclusion[d],listC[d1]).correlation)
    smallCorMatrix.append(corLine)

print('done')
        


# In[415]:


df.loc[0].tolist()


# In[435]:


hugeCorMatrix = []
hugeCorMatrix.append([])
hugeCorMatrix.append([])
hugeCorMatrix.append([])

staticval0 = listForConclusion[0][:-1]
staticval1 = listForConclusion[1][:-1]
staticval2 = listForConclusion[2][:-1]

for l in range(1,len(df)):
    changeVal = df.loc[l].tolist()[1:]
    if len(changeVal) == len(staticval0):
        hugeCorMatrix[0].append(stts.spearmanr(staticval0,changeVal).correlation)
        hugeCorMatrix[1].append(stts.spearmanr(staticval1,changeVal).correlation)
        hugeCorMatrix[2].append(stts.spearmanr(staticval2,changeVal).correlation)
    if l % 1000 == 0:
        print('l: ' + str(l) )
print ('done')


# In[444]:


ax = sns.heatmap(hugeCorMatrix, annot=True, fmt="f")


# In[445]:


ax = sns.heatmap(hugeCorMatrix)


# In[436]:


fig23 = plt.figure()
fig23 = plt.matshow(hugeCorMatrix)
plt.colorbar()


# In[377]:


fig22 = plt.figure()
fig22 = plt.matshow(smallCorMatrix)
plt.colorbar()


# In[446]:


ax2 = sns.heatmap(smallCorMatrix)

