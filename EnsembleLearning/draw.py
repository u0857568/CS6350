import pandas as pd
import matplotlib.pyplot as plt 

headers = ['train error','test error']
df = pd.read_csv('randforests_featsz6.csv', names=headers)
print(df)

#for x in range(100):

df.plot(y=['train error','test error'],title='randforests_6')
plt.show()
