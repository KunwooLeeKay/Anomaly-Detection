import pandas as pd

data = pd.read_excel(r"C:\Users\user\Desktop\LearnPython\Lab\MachineLearningData_.xlsx")

feature = ['제조국가', '사용년수', 'C', 'D', 'E', '진동 횟수', '최대 기계압력', '최소 기계압력', '기계 길이',\
                        '기계 무게', '특수 부품 유무', '특수 부품 저항', '일반 부품 저항', '기계 마력', 'O', 'P',\
                        '기계 부품 임피던스', 'P/J', 'S', 'T', 'U', '기계 평균 압력', 'R/J', 'M*J', 'L*J',\
                        'L/J']

x데이터 = data[feature]
y데이터 = data[['고장 유무']]

from imblearn.over_sampling import SMOTE

x데이터, y데이터 = SMOTE().fit_resample(x데이터, y데이터)

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh'), # 히든 레이어 만드는 문법, 64는 노드 수
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation = 'sigmoid') # 마지막 노드 수는 원하는 결과 개수에 따라 다르다. 지금은 확률 한개니까 하나.
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit( x데이터, y데이터, epochs = 100 )
