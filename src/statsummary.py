import pandas as pd

#url ="https://goo.gl/dBdBiA"
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas','pres', 'skin','test','mass','pedi','age','class']

df  = pd.read_csv(url,names=names)
description = df.describe()

print(description)