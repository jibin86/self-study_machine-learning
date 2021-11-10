# -*- coding: utf-8 -*-
"""특성공학과 규제.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TcE6MLV2x7XYLkmWNplmPLB3c0aH-ZzH
"""

# 작성 날짜: 2021.09.05.일
# 프로그램 개요: 다중회귀(릿지회귀, 라쏘회귀) - 농어의 무게 예측하기

#데이터 준비

#판다스의 데이터 프레임으로 바꾸기
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full) #특성

import numpy as np
#타깃
perch_weight = np.array([  5.9,  32.0,  40.0,  51.5,  70.0, 100.0,  78.0,  80.0,  85.0,  85.0, 
                         110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0, 
                         150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0,
                         300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0, 
                         685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0,
                         820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0])

#훈련, 테스트 세트 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

#특성 공학 (변환기)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
print(poly.get_feature_names())
test_poly = poly.transform(test_input)

# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

#특성 더 늘려보기
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) #샘플보다 특성이 많음

#다시 훈련
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# --> 심각한 과대적합! => 규제 필요

#규제 (가중치 줄이기)

#특성 정규화 하기
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#릿지 회귀 (선형회귀에 L2규제를 적용한 회귀)
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

#alpha(규제 정도)에 따른 릿지회귀의 score 구하기
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  #릿지 모델 만들기
  ridge = Ridge(alpha=alpha)
  #훈련
  ridge.fit(train_scaled, train_target)
  #score 저장
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

#그리기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

#최적 alpha 값
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

#라쏘 회귀 (선형 회귀에 L1규제를 적용한 회귀)

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

#alpha(규제 정도)에 따른 라쏘 회귀의 score 구하기

train_score = []
test_score = []

for alpha in alpha_list:
  lasso = Lasso(alpha=alpha, max_iter=10000) #max_iter=10000으로 함으로써 반복횟수 늘림
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))

#그리기
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
print(train_score)
print(test_score)

#최적 alpha 값
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

#가중치가 0인 특성 개수
print(np.sum(lasso.coef_ == 0)) #--> 특성 55개 중 15개만 사용함