import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
def readFile(filePath):
    file = pd.read_csv(filePath,encoding="utf-8")

    label = file[file.columns[-1]]
    content = file[file.columns[-2]]
    return content,label

