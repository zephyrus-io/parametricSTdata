from functions import *
import pandas as pd
import numpy as np

data = pd.read_csv('oneDeviceOneDay.tsv', sep='\t')
data = data[['deviceTime', 'lastSeen', 'point','accuracy']]
p = [coordParser(x) for x in data.point]
x = [float(i[0]) for i in p]
y = [float(i[1]) for i in p]
data['x'] = x
data['y'] = y
del data['point']
data.head()

#sort data by devicetime
dd = data.sort_values(by='deviceTime', axis=0)

h = STprep(dd)
h.head()

control = []
for index, row in h.iterrows():
    c = bezierSolver(row)
    control.append(c)

print('done')

h['bezContrPts'] = control
from shapely.geometry import LineString

h.head()

lines = [LineString(i) for i in h['bezContrPts']]
l = [dumps(i) for i in lines]
l[0]
print(lines[0])
h['Linestring'] = lines

h.head()
h.to_csv('bezierControlPoints.tsv', index=False, sep='\t')

geometry = h[['Linestring','i','ij']]
geometry.head()
geometry.to_csv('beziergeometry.csv', index=False,sep='|', header=None)
line = LineString(h.iloc[0]['bezContrPts'])
