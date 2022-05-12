
# In[1]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('gold_rate_history.csv')

x=df.drop(['Date'],axis=1)
y=df['Date']





from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.20, random_state = 42 )



from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x,y)





Y_pred = neigh.predict(X_test)





pickle.dump(neigh, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
#print(model.predict(sc.transform(np.array([[20,40]]))))


