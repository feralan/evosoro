import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def calcInterDistance(weightMatrix, additionalData):
    calculatedValues = [];
    res = 0;

    for index in range(0, len(additionalData.pressures)):
        calculatedValues.append(np.sum(np.multiply(additionalData.pressures[index], weightMatrix)))

    for indIndex in range(0, len(additionalData.pressures)):
        for index in range(0, len(additionalData.pressures)):
            if additionalData.labels[indIndex] != additionalData.labels[index]:
                res += np.abs(calculatedValues[indIndex] - calculatedValues[index])

    return res

def calcIntraDistance(weightMatrix, additionalData):
    calculatedValues = [];
    res = 0;

    for index in range(0, len(additionalData.pressures)):
        calculatedValues.append(np.sum(np.multiply(additionalData.pressures[index], weightMatrix)))

    for indIndex in range(0, len(additionalData.pressures)):
        for index in range(0, len(additionalData.pressures)):
            if additionalData.labels[indIndex] == additionalData.labels[index]:
                res += np.abs(calculatedValues[indIndex] - calculatedValues[index])

    return res

def silhouetteCoeff(weightMatrix, additionalData):
    weightMatrix = np.transpose(weightMatrix)
    sil = 0
    
    calculatedValues = [];
    for index in range(0, len(additionalData.pressures)):
        calculatedValues.append(np.sum(np.multiply(additionalData.pressures[index], weightMatrix)))

    weightedPressures = np.multiply(np.array(additionalData.pressures), weightMatrix)
    try:
        clf = LinearDiscriminantAnalysis()
        clf.fit(weightedPressures, additionalData.labels)    
    except:
        clf = LinearDiscriminantAnalysis(solver = 'lsqr')
        clf.fit(weightedPressures, additionalData.labels)
    isCorrectLabel = []
    for selectedEle in range(0, len(additionalData.pressures)):
        estLbl = clf.predict([weightedPressures[selectedEle]])
        isCorrectLabel.append(estLbl[0])
        if estLbl[0] != additionalData.labels[selectedEle]:
            sil += 0
            continue
        else:
            intra = 0
            inter = 0
            intraCount = 0
            interCount = 0
            for otherEle in range(0, len(additionalData.pressures)):
                if additionalData.labels[selectedEle] == additionalData.labels[otherEle]:
                    intra += np.abs(calculatedValues[selectedEle] - calculatedValues[otherEle])
                    intraCount += 1
                else:
                    inter += np.abs(calculatedValues[selectedEle] - calculatedValues[otherEle])
                    interCount += 1
            inter = inter/interCount
            intra = intra/intraCount
            if (inter - intra) != 0:
                sil += ((inter - intra) / (np.amax([inter, intra])))
            else:
                sil += 0

    #sil=sil/len(additionalData.pressures)

    return sil
