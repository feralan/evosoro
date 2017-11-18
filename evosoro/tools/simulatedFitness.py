import numpy as np

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
    calculatedValues = [];
    sil = 1

    for index in range(0, len(additionalData.pressures)):
        calculatedValues.append(np.sum(np.multiply(additionalData.pressures[index], weightMatrix)))

    for selectedEle in range(0, len(additionalData.pressures)):
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
            sil *= ((inter - intra) / (np.amax([inter, intra])))
        else:
            sil = 0

    #sil=sil/len(additionalData.pressures)

    return sil
