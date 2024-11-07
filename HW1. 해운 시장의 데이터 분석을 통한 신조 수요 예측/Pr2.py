import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# 엑셀 파일 경로 설정
file_path = 'Order estimation.xlsx'

# 엑셀 파일에서 데이터 읽기 (첫 번째 행이 데이터의 일부이므로 header=None)
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

# C26~C103, D26~D103, F26~F103의 값만 선택
selected_data = df.loc[25:103, [2, 3, 5]]  # 인덱스는 0부터 시작하므로 25는 C26에 해당

# C, D, F 열의 값을 각각 저장
x1 = selected_data[2].values
x2 = selected_data[3].values
Y = selected_data[5].values

# 독립 변수들을 하나의 배열로 결합
X = np.column_stack((x1, x2))

# 회귀 모델 생성 및 학습
model = LinearRegression(fit_intercept=False)  # Bias가 없으므로 fit_intercept=False 설정
model.fit(X, Y)

# 회귀 계수 출력
a5, a6 = model.coef_
print(f"a_1: {a5}, a_2: {a6}")
print(f"회귀 방정식: Y = {a5} * x1 + {a6} * x2")

# 예측값 계산
Y_pred = model.predict(X)

# 1. 실제 값과 예측 값 비교 그래프
plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(Y)), Y, color='blue', label='real (Y)')
plt.plot(np.arange(len(Y_pred)), Y_pred, color='red', label='predict (Y_pred)')
plt.xlabel('Data index')
plt.ylabel('value')
plt.title('Difference between pred and real')
plt.legend()
plt.grid(True)
plt.show()

# 2. x1의 영향
plt.figure(figsize=(10, 6))
plt.scatter(x1, Y, color='blue', label='real (Y)')
plt.plot(x1, a5 * x1, color='red', label='a5 * x1')
plt.xlabel('x1')
plt.ylabel('Y')
plt.title('Effect of x1 on Y')
plt.legend()
plt.grid(True)
plt.show()

# 3. x2의 영향
plt.figure(figsize=(10, 6))
plt.scatter(x2, Y, color='blue', label='real (Y)')
plt.plot(x2, a6 * x2, color='red', label='a6 * x2')
plt.xlabel('x2')
plt.ylabel('Y')
plt.title('Effect of x2 on Y')
plt.legend()
plt.grid(True)
plt.show()

# 4. 3D 그래프로 x1, x2, Y의 관계 표현
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, Y, c='blue', label='real (Y)')
x1_pred = np.linspace(min(x1), max(x1), 100)
x2_pred = np.linspace(min(x2), max(x2), 100)
X1_pred, X2_pred = np.meshgrid(x1_pred, x2_pred)
Y_pred_surface = a5 * X1_pred + a6 * X2_pred
ax.plot_surface(X1_pred, X2_pred, Y_pred_surface, alpha=0.5, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Y')
ax.set_title('3D relationship between x1, x2, and Y')
ax.legend()
plt.show()