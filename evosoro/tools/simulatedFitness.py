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
