import pandas as pd

cp = pd.read_csv('bezierControlPoints.tsv', sep='\t')
cp.head()
tpoints = pd.read_csv('oneDeviceOneday.tsv', sep='\t')
tpoints.head()

tpoints['tag'] = 'NA'

for index, row in cp.iterrows():

    tpoints.set_value(list(range(row.i,row.ij+1)),'tag',str(row.i)+'_'+str(row.ij))

print('done')

tpoints.head(20)

output = tpoints[['deviceTime', 'x','y','tag','rank']]
output.to_csv('taggedPoints.tsv', sep='\t', index=False)