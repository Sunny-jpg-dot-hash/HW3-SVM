import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.svm import SVC

# 生成方形分布的資料點
def generate_square_data(num_points, side_length):
    half_side = side_length / 2
    x1 = np.random.uniform(-half_side, half_side, num_points)
    x2 = np.random.uniform(-half_side, half_side, num_points)
    
    # 方形範圍內中心方形為 0，邊緣部分為 1
    labels = np.where((np.abs(x1) < half_side / 2) & (np.abs(x2) < half_side / 2), 0, 1)
    return x1, x2, labels

# 計算第三維度以增加 3D 視覺效果
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# 訓練 SVM 並返回分隔平面的係數和截距
def train_svm(x1, x2, x3, labels):
    X = np.column_stack((x1, x2, x3))
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, labels)
    return clf.coef_[0], clf.intercept_

# 繪製 3D 圖表，包括資料點和分隔平面
def plot_3d(x1, x2, x3, labels, coef, intercept):
    fig = go.Figure()

    # 3D 散點圖顯示資料點
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=x3, mode='markers',
        marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.7),
        name="Data Points"
    ))

    # 計算並繪製分隔平面
    xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                         np.linspace(min(x2), max(x2), 10))
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Blues', opacity=0.5, name="Decision Boundary"))

    # 設定 3D 圖表佈局
    fig.update_layout(
        scene=dict(
            xaxis_title='x1', yaxis_title='x2', zaxis_title='x3',
            xaxis=dict(nticks=10, range=[min(x1), max(x1)]),
            yaxis=dict(nticks=10, range=[min(x2), max(x2)]),
            zaxis=dict(nticks=10)
        ),
        title='3D Scatter Plot with SVM Decision Boundary'
    )
    return fig

# 主程序
if __name__ == "__main__":
    st.title("2D SVM with Streamlit Deployment and 3D Plot (Square Distribution)")

    # 設定參數
    num_points = st.slider("Number of Points", min_value=100, max_value=1000, value=600, step=100)
    side_length = st.slider("Square Side Length for Labels", min_value=2.0, max_value=20.0, value=10.0)

    # 生成資料並計算第三維度
    x1, x2, labels = generate_square_data(num_points, side_length)
    x3 = gaussian_function(x1, x2)

    # 訓練 SVM 並獲得分隔平面
    coef, intercept = train_svm(x1, x2, x3, labels)

    # 繪製 3D 圖表
    fig = plot_3d(x1, x2, x3, labels, coef, intercept)
    st.plotly_chart(fig)
