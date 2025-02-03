import pandas as pd
d=pd.read_csv('data.csv')
d['years'] = 2*d['years']
d.to_csv('data.csv',index=False)