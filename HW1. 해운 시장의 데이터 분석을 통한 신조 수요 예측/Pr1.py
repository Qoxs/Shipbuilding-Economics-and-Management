import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# 엑셀 파일 경로 설정 (필요시 수정)
file_path = 'Freight estimation.xlsx'

# 엑셀 파일에서 데이터 읽기 (첫 번째 행이 데이터의 일부이므로 header=None)
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

selected_data = df.loc[8:184, [3, 5, 6]]

# D열과 F열의 값을 각각 저장하고 숫자형으로 변환
x = pd.to_numeric(selected_data[3], errors='coerce').values.reshape(-1, 1)  # D열 (Utilization rate)
y = pd.to_numeric(selected_data[5], errors='coerce').values  # F열 (3 month Freight index)
z = pd.to_numeric(selected_data[6], errors='coerce').values  # G열 (6 month Freight index)

# NaN 값 제거
mask = ~np.isnan(x.flatten()) & ~np.isnan(y) & ~np.isnan(z)
x = x[mask]
y = y[mask]
z = z[mask]

# 예측을 위한 x 값 범위 설정 (더 많은 x 값을 생성하여 곡선을 부드럽게)
x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)

# 3개월 

# 선형 회귀 (Linear Regression)
linear_model_y = LinearRegression()
linear_model_y.fit(x, y)
print("선형 회귀 결과:")
print(f"방정식: y = {linear_model_y.intercept_:.4f} + {linear_model_y.coef_[0]:.4f}x")
print(f"R-squared: {r2_score(y, linear_model_y.predict(x)):.4f}")
print(f"MSE: {mean_squared_error(y, linear_model_y.predict(x)):.4f}\n")

# 2차 다항 회귀 (Quadratic Regression)
poly_features_2_y = PolynomialFeatures(degree=2, include_bias=False)
x_poly_2_y = poly_features_2_y.fit_transform(x)
poly_model_2_y = LinearRegression()
poly_model_2_y.fit(x_poly_2_y, y)
print("2차 다항 회귀 결과:")
print(f"방정식: y = {poly_model_2_y.intercept_:.4f} + {poly_model_2_y.coef_[0]:.4f}x + {poly_model_2_y.coef_[1]:.4f}x^2")
print(f"R-squared: {r2_score(y, poly_model_2_y.predict(x_poly_2_y)):.4f}")
print(f"MSE: {mean_squared_error(y, poly_model_2_y.predict(x_poly_2_y)):.4f}\n")

# 3차 다항 회귀 (Cubic Regression)
poly_features_3_y = PolynomialFeatures(degree=3, include_bias=False)
x_poly_3_y = poly_features_3_y.fit_transform(x)
poly_model_3_y = LinearRegression()
poly_model_3_y.fit(x_poly_3_y, y)
print("3차 다항 회귀 결과:")
print(f"방정식: y = {poly_model_3_y.intercept_:.4f} + {poly_model_3_y.coef_[0]:.4f}x + {poly_model_3_y.coef_[1]:.4f}x^2 + {poly_model_3_y.coef_[2]:.4f}x^3")
print(f"R-squared: {r2_score(y, poly_model_3_y.predict(x_poly_3_y)):.4f}")
print(f"MSE: {mean_squared_error(y, poly_model_3_y.predict(x_poly_3_y)):.4f}\n")


# 예측값 계산
y_pred_linear_y = linear_model_y.predict(x_range)
y_pred_poly_2_y = poly_model_2_y.predict(poly_features_2_y.transform(x_range))
y_pred_poly_3_y = poly_model_3_y.predict(poly_features_3_y.transform(x_range))

# 6개월

# 선형 회귀 (Linear Regression)
linear_model_z = LinearRegression()
linear_model_z.fit(x, z)
print("선형 회귀 결과:")
print(f"방정식: z = {linear_model_z.intercept_:.4f} + {linear_model_z.coef_[0]:.4f}x")
print(f"R-squared: {r2_score(z, linear_model_z.predict(x)):.4f}")
print(f"MSE: {mean_squared_error(z, linear_model_z.predict(x)):.4f}\n")

# 2차 다항 회귀 (Quadratic Regression)
poly_features_2_z = PolynomialFeatures(degree=2, include_bias=False)
x_poly_2_z = poly_features_2_z.fit_transform(x)
poly_model_2_z = LinearRegression()
poly_model_2_z.fit(x_poly_2_z, z)
print("2차 다항 회귀 결과:")
print(f"방정식: z = {poly_model_2_z.intercept_:.4f} + {poly_model_2_z.coef_[0]:.4f}x + {poly_model_2_z.coef_[1]:.4f}x^2")
print(f"R-squared: {r2_score(z, poly_model_2_z.predict(x_poly_2_z)):.4f}")
print(f"MSE: {mean_squared_error(z, poly_model_2_z.predict(x_poly_2_z)):.4f}\n")

# 3차 다항 회귀 (Cubic Regression)
poly_features_3_z = PolynomialFeatures(degree=3, include_bias=False)
x_poly_3_z = poly_features_3_z.fit_transform(x)
poly_model_3_z = LinearRegression()
poly_model_3_z.fit(x_poly_3_z, z)
print("3차 다항 회귀 결과:")
print(f"방정식: z = {poly_model_3_z.intercept_:.4f} + {poly_model_3_z.coef_[0]:.4f}x + {poly_model_3_z.coef_[1]:.4f}x^2 + {poly_model_3_z.coef_[2]:.4f}x^3")
print(f"R-squared: {r2_score(z, poly_model_3_z.predict(x_poly_3_z)):.4f}")
print(f"MSE: {mean_squared_error(z, poly_model_3_z.predict(x_poly_3_z)):.4f}\n")

# 예측값 계산
y_pred_linear_z = linear_model_z.predict(x_range)
y_pred_poly_2_z = poly_model_2_z.predict(poly_features_2_z.transform(x_range))
y_pred_poly_3_z = poly_model_3_z.predict(poly_features_3_z.transform(x_range))



plt.figure(figsize=(12, 8))
plt.scatter(x, y, color='blue', label='Real data (3 month)')
plt.plot(x_range, y_pred_linear_y, color='red', label='Linear regression')
plt.plot(x_range, y_pred_poly_2_y, color='green', label='Quadratic regression')
plt.plot(x_range, y_pred_poly_3_y, color='orange', label='Cubic regression')
plt.xlabel('Utilization rate')
plt.ylabel('Freight index (3 month)')
plt.title('Relation between Utilization rate and Freight index (3 month)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
plt.scatter(x, z, color='blue', label='Real data (6 month)')
plt.plot(x_range, y_pred_linear_z, color='red', label='Linear regression')
plt.plot(x_range, y_pred_poly_2_z, color='green', label='Quadratic regression')
plt.plot(x_range, y_pred_poly_3_z, color='orange', label='Cubic regression')
plt.xlabel('Utilization rate')
plt.ylabel('Freight index (6 month)')
plt.title('Relation between Utilization rate and Freight index (z)')
plt.legend()
plt.grid(True)
plt.show()


# ML로 최적 회귀 찾기
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 엑셀 파일 경로 설정
file_path = 'Freight estimation.xlsx'

# 엑셀 파일에서 데이터 읽기 (첫 번째 행이 데이터의 일부이므로 header=None)
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

selected_data = df.loc[8:184, [3, 5, 6]]

# D열과 F열의 값을 각각 저장하고 숫자형으로 변환
x = pd.to_numeric(selected_data[3], errors='coerce').values.reshape(-1, 1)  # D열 (Utilization rate)
y = pd.to_numeric(selected_data[5], errors='coerce').values  # F열 (Freight index)

# NaN 값 제거
mask = ~np.isnan(x.flatten()) & ~np.isnan(y)
x = x[mask]
y = y[mask]

# 예측을 위한 x 값 범위 설정 (더 많은 x 값을 생성하여 곡선을 부드럽게 함)
x_range = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)

# 최대 50차까지 다항 회귀를 수행하고 MSE를 계산 (각 epoch에서 평균 MSE)
min_mse = float('inf')
best_degree = 0
mse_list = []

epochs = 500  # Epoch 수 설정

for degree in range(1, 51):
    mse_epoch_list = []  # 각 epoch에서 MSE 저장
    
    for epoch in range(epochs):
        # 다항식 변환
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly_features.fit_transform(x)
        
        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(x_poly, y)
        
        # 예측값 계산
        y_pred = model.predict(x_poly)
        
        # MSE 계산
        mse_epoch_list.append(mean_squared_error(y, y_pred))
    
    # 각 degree에 대한 평균 MSE 계산
    avg_mse = np.mean(mse_epoch_list)
    mse_list.append(avg_mse)
    
    # 최적의 차수 찾기
    if avg_mse < min_mse:
        min_mse = avg_mse
        best_degree = degree

# 최적의 차수로 다시 모델 학습 및 예측
poly_features_best = PolynomialFeatures(degree=best_degree, include_bias=False)
x_poly_best = poly_features_best.fit_transform(x)

model_best = LinearRegression()
model_best.fit(x_poly_best, y)

# x_range에 대한 예측값 계산 (부드러운 곡선을 그리기 위해)
y_range_pred_best = model_best.predict(poly_features_best.transform(x_range))

# 결과 출력
print(f"가장 적합한 차수: {best_degree}")
print(f"최소 MSE: {min_mse}")

# MSE 그래프 그리기 (차수에 따른 MSE 변화)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), mse_list, marker='o', color='blue')
plt.title('Polynomial Degree vs Average MSE over 500 epochs')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.show()

# 최적 차수에 대한 회귀선 그래프 그리기
plt.figure(figsize=(12, 8))
plt.scatter(x, y, color='blue', label='Real data')
plt.plot(x_range, y_range_pred_best, color='red', label=f'Best Polynomial Regression (degree={best_degree})')
plt.xlabel('Utilization rate')
plt.ylabel('Freight index')
plt.title(f'Best Polynomial Regression with degree {best_degree}')
plt.legend()
plt.grid(True)
plt.show()
'''