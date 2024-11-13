import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 1. 生成 300 個隨機變量 X，範圍在 [0, 1000]
np.random.seed(42)
X = np.random.randint(0, 1001, 300)

# 2. 創建二元分類標籤 Y，根據條件 (500 < X < 800) 為 1，否則為 0
Y = np.where((X > 500) & (X < 800), 1, 0)

# 3. 分割數據集為訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=42)

# 4. 訓練邏輯回歸模型
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# 5. 訓練支持向量機 (SVM) 模型
svm = SVC(probability=True)  # 設定 probability=True 以獲得預測概率
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# 6. 視覺化結果
plt.figure(figsize=(15, 6))

# 圖表 1：顯示 Logistic Regression 的預測結果
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='真實標籤')
plt.scatter(X_test, y1, color='blue', marker='x', label='Logistic Regression 預測')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y 與 Logistic Regression 預測')
plt.legend()

# 邏輯回歸的決策邊界
x_boundary = np.linspace(0, 1000, 300)
y_boundary = logreg.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='blue', linestyle='--', label='Logistic Regression 邊界')
plt.legend()

# 圖表 2：顯示 SVM 的預測結果
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='真實標籤')
plt.scatter(X_test, y2, color='green', marker='s', label='SVM 預測')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y 與 SVM 預測')
plt.legend()

# SVM 的決策邊界
y_boundary = svm.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='SVM 邊界')
plt.legend()

plt.tight_layout()
plt.show()
