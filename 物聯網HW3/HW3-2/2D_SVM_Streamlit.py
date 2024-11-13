import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.svm import LinearSVC

# 生成圓形分布的資料點
def generate_circular_data(num_points, radius, center=(0, 0)):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, num_points))
    x1 = center[0] + radii * np.cos(angles)
    x2 = center[1] + radii * np.sin(angles)
    labels = np.where(radii < radius / 2, 0, 1)  # 半徑小於一半為 0，其餘為 1
    return x1, x2, labels

# 計算第三維度以增加 3D 視覺效果
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# 訓練 SVM 並返回分隔平面的係數和截距
def train_svm(x1, x2, x3, labels):
    X = np.column_stack((x1, x2, x3))
    clf = LinearSVC(random_state=0, max_iter=10000, dual=False)
    clf.fit(X, labels)
    return clf.coef_[0], clf.intercept_

# 繪製 3D 圖表
def plot_3d(x1, x2, x3, labels, coef, intercept):
    fig = go.Figure()

    # 3D 散點圖顯示資料點
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=x3, mode='markers',
        marker=dict(size=5, color=['blue' if label == 0 else 'red' for label in labels], opacity=0.7),
        name="Data Points"
    ))

    # 分隔平面
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        opacity=0.5,
        showscale=False,
        name="Separating Hyperplane"
    ))

    # 圖表佈局
    fig.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='x3'),
                      title='3D 圓形分布資料的 SVM 分隔超平面')
    return fig

# 主程序
if __name__ == "__main__":
    st.title("2D SVM with Streamlit Deployment and 3D Plot (Circular Distribution)")

    # 設定固定參數
    num_points = 600  # 固定資料點數量
    radius = 8.0      # 固定圓形分布半徑

    # 生成圓形分布資料點
    x1, x2, labels = generate_circular_data(num_points, radius)

    # 計算第三維度
    x3 = gaussian_function(x1, x2)

    # 訓練 SVM 模型
    coef, intercept = train_svm(x1, x2, x3, labels)

    # 繪製 3D 圖表
    fig = plot_3d(x1, x2, x3, labels, coef, intercept)
    st.plotly_chart(fig)
