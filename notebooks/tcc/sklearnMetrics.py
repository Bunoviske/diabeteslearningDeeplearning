from typing import List, Tuple
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

class Metrics():
    
    def __init__(self, classes: List[str], classesToRemoveIdx:List[int]) -> None:
        self.classesToRemoveIdx = classesToRemoveIdx
        self.classesIdx = np.arange(0, len(classes)).tolist()
        self.classesIdx = [idx for idx in self.classesIdx if idx not in classesToRemoveIdx]
        self.classes = [classes[idx] for idx in self.classesIdx]
        
    def _removeClasses(self, y_pred: np.ndarray, y_true:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        
        for classToRemove in self.classesToRemoveIdx:
            indexes = np.where(y_true == classToRemove)[0]
            y_true = np.delete(y_true, indexes)
            y_pred = np.delete(y_pred, indexes)
            
        return y_pred, y_true
        
    def getAccuracy(self, y_pred: np.ndarray, y_true:np.ndarray) -> float:
        y_pred, y_true = self._removeClasses(y_pred, y_true)
        return metrics.accuracy_score(y_true, y_pred)
            
    def getClassificationReport(self, y_pred: np.ndarray, y_true:np.ndarray):
        return metrics.classification_report(y_true, y_pred, labels=self.classesIdx,\
                                             target_names=self.classes,zero_division=1)
    
    def get_f1Score(self, y_pred: np.ndarray, y_true:np.ndarray, average='micro'):
        return metrics.f1_score(y_true, y_pred, labels=self.classesIdx, average=average,zero_division=1)

    def getConfusionMatrix(self, y_pred: np.ndarray, y_true:np.ndarray, normalize:str = "all", plot: bool = True):
        cm = metrics.confusion_matrix(y_true, y_pred, labels=self.classesIdx, normalize=normalize)
        if plot:
            disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
            print("plot takes too long....")
            disp.plot()
        return cm

    def plotConfusionMatrix(self, confusionMatrix: np.ndarray, codes: List[str]):
        df_cm = pd.DataFrame(confusionMatrix, index = codes[1:], columns = codes[1:])
        plt.figure(figsize = (40,40))
        plt.tight_layout()
        sn.set(font_scale=0.5) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
        plt.savefig("cm.png")
        plt.show()

    def mostConfused(self, confusionMatrix, dataLoader):
        np.fill_diagonal(confusionMatrix, 0)
        auxiliarMatrix = confusionMatrix
        mostConfused = []
        numberOfConfusions = 5

        for z in range(0, numberOfConfusions): #beeing numberOfConfusions the variable which defines how many entries from the confusionMatrix
            i = np.where(auxiliarMatrix==np.max(auxiliarMatrix))[0][0]
            j = np.where(auxiliarMatrix==np.max(auxiliarMatrix))[1][0]
            res = [dataLoader.vocab[i], dataLoader.vocab[j], auxiliarMatrix[i][j]]
            mostConfused.append(res)
            auxiliarMatrix[i][j] = 0
            

        return mostConfused