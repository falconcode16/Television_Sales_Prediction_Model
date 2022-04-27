import warnings

import streamlit as st 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
sns.set()
from sklearn.datasets import load_boston
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


header = st.container()
dataset = st.container()
features =st.container()
model_training = st.container()


with header:
    st.title('Television Sales Prediction Model ')
    st.text('Business improvization model')
with dataset:
    st.header('Dataset')
    df=pd.read_csv('data/advertising.csv')
    if st.checkbox("Show dataset"):
       st.dataframe(df.head(20))
    st.write(df.describe())
    st.write(df.isnull().sum()*100/df.shape[0])
    
    
fig, ax = plt.subplots()
sns.heatmap(df.corr(),cmap="YlGnBu", annot = True)
st.write(fig)

"""
### Split the dataset
"""
left_column, right_column = st.columns(2)

# test size
test_size = left_column.number_input(
				'Validation-dataset size (rate: 0.0-1.0):',
				min_value=0.0,
				max_value=1.0,
				value=0.2,
				step=0.1,
				 )

# random_seed
random_seed = right_column.number_input('Set random seed (0-):',
							  value=0, step=1,
							  min_value=0)

# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
	df["TV"], 
	df['Sales'], 
	test_size=test_size, 
	random_state=random_seed
	)

X_train = np.array(X_train).reshape(-1,1)
st.write(X_train.shape)
Y_train = np.array(Y_train).reshape(-1,1)
st.write(Y_train.shape)

import numpy as np
from sklearn.linear_model import LinearRegression

lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)

fig, ax = plt.subplots()
plt.scatter(X_train, Y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
st.write(fig)

st.header("Model Evaluations")

Y_train_pred = lr.predict(X_train)
res = (Y_train - Y_train_pred)

fig, ax = plt.subplots()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
st.pyplot(fig)
st.header("Predictions")
st.write(X_test.shape)

X_test = np.array(X_test).reshape(-1,1)

Y_pred= lr.predict(X_test)

st.write(Y_pred)
fig, ax = plt.subplots()
plt.scatter(X_test, Y_test)
plt.plot(X_test, 6.948 + 0.054*X_test, 'r')
st.write(fig)
"""
## Show the result
### Check R2 socre
"""
R2 = r2_score(Y_test, Y_pred)
st.write(f'R2 score: {R2:.2f}')


"""
### Plot the result
"""
left_column, right_column = st.columns(2)
show_train = left_column.radio(
				'Show the training dataset:', 
				('Yes','No')
				)
show_val = right_column.radio(
				'Show the Testing dataset:', 
				('Yes','No')
				)

# default axis range
y_max_train = max([max(Y_train), max(Y_train_pred)])
y_max_test = max([max(Y_test), max(Y_pred)])
y_max = int(max([y_max_train, y_max_test])) 

# interactive axis range
left_column, right_column = st.columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
	plt.scatter(Y_train, Y_train_pred,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
	plt.scatter(Y_test, Y_pred,lw=0.1,color="b",label="Test data")
plt.xlabel("TV",fontsize=8)
plt.ylabel("Sales",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)