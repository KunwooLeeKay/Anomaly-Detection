import pandas as pd

data = pd.read_csv(r'C:\Users\user\Desktop\LearnPython\고장진단 프로젝트\Restart\gpascore.csv')
print(data.isnull().sum())

data = data.dropna()
print(data.isnull().sum())

# data.fillna() # 이거 하면 결손데이터 채우기 

x데이터 = data[['gre', 'gpa', 'rank']]
y데이터 = data[['admit']]

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh'), # 히든 레이어 만드는 문법, 64는 노드 수
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid') # 마지막 노드 수는 원하는 결과 개수에 따라 다르다. 지금은 확률 한개니까 하나.
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit( x데이터, y데이터, epochs = 1000 )

예측값 = model.predict([ [750, 3.70, 3], [400, 2.2, 1] ])
print(예측값)