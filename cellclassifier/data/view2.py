import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

# 定义数据
data = {
    'Fold 1': {
        'LX-2': [-1.298, -1.175, -1.679, -1.650, -0.997],
        'Hep3B': [0.127, -0.463, -0.192, -0.193, -0.965],
        'HepG2': [-0.270, -0.148, 0.361, 0.567, 0.562],
        'Huh-7': [0.611, 0.699, 0.800, 0.651, 0.816],
        'MHCC97H': [0.828, 1.168, 0.722, 0.647, 0.740]
    },
    'Fold 2': {
        'LX-2': [-1.343, -1.161, -1.681, -1.648, -1.025],
        'Hep3B': [0.073, -0.506, -0.192, -0.261, -0.972],
        'HepG2': [-0.131, -0.055, 0.364, 0.573, 0.575],
        'Huh-7': [0.678, 0.621, 0.745, 0.735, 0.866],
        'MHCC97H': [0.853, 1.032, 0.701, 0.727, 0.587]
    },
    'Fold 3': {
        'LX-2': [-1.333, -1.186, -1.621, -1.641, -0.982],
        'Hep3B': [0.192, -0.404, -0.236, -0.237, -1.026],
        'HepG2': [-0.214, -0.122, 0.396, 0.582, 0.573],
        'Huh-7': [0.707, 0.581, 0.769, 0.735, 0.796],
        'MHCC97H': [0.752, 1.187, 0.816, 0.663, 0.689]
    },
    'Fold 4': {
        'LX-2': [-1.353, -1.169, -1.638, -1.623, -1.008],
        'Hep3B': [0.134, -0.517, -0.213, -0.246, -1.066],
        'HepG2': [-0.421, 0.047, 0.309, 0.647, 0.565],
        'Huh-7': [0.633, 0.509, 0.771, 0.683, 0.845],
        'MHCC97H': [0.754, 1.222, 0.734, 0.671, 0.728]
    },
    'Fold 5': {
        'LX-2': [-1.363, -1.224, -1.603, -1.647, -1.017],
        'Hep3B': [0.161, -0.499, -0.172, -0.307, -1.044],
        'HepG2': [-0.232, -0.086, 0.322, 0.473, 0.527],
        'Huh-7': [0.606, 0.546, 0.830, 0.752, 0.853],
        'MHCC97H': [0.766, 1.145, 0.745, 0.627, 0.673]
    }
}

# 准备数据用于PCA
all_centroids = []
labels = []
fold_labels = []
cell_types = []

fold_names = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
cell_types_list = ['LX-2', 'Hep3B', 'HepG2', 'Huh-7', 'MHCC97H']
markers = ['o', 's', '^', 'D', 'v']  # 圆形, 方形, 三角形, 菱形, 倒三角形
colors = ['blue', 'green', 'red', 'purple', 'orange']  # 为每个细胞类型分配颜色

for i, fold in enumerate(fold_names):
    for cell_type in cell_types_list:
        centroid = data[fold][cell_type]
        all_centroids.append(centroid)
        labels.append(f'{cell_type} ({fold})')
        fold_labels.append(i)  # 记录属于哪一折
        cell_types.append(cell_type)

# 转换为numpy数组
X = np.array(all_centroids)

# 应用PCA降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"总解释方差: {sum(pca.explained_variance_ratio_):.3f}")

# 创建图形 - 使用更大的尺寸
plt.figure(figsize=(14, 12))

# 为每个细胞类型和折数绘制点 - 增加点的大小
for i, cell_type in enumerate(cell_types_list):
    for j, fold in enumerate(fold_names):
        # 找到对应的索引
        indices = [idx for idx, (ct, fl) in enumerate(zip(cell_types, fold_labels))
                   if ct == cell_type and fl == j]

        if indices:
            x = X_pca[indices, 0]
            y = X_pca[indices, 1]
            plt.scatter(x, y, marker=markers[j], color=colors[i],
                        s=150, alpha=0.8, edgecolor='white', linewidth=1.5)

# 添加图例
from matplotlib.lines import Line2D

# 创建颜色图例（细胞类型）- 增加图例标记大小
color_legend_elements = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=colors[i], markersize=14,
                                label=cell_types_list[i], markeredgewidth=1.5)
                         for i in range(len(cell_types_list))]

# 创建形状图例（折数）- 增加图例标记大小
marker_legend_elements = [Line2D([0], [0], marker=markers[i], color='w',
                                 markerfacecolor='black', markersize=14,
                                 label=f'Fold {i + 1}', markeredgewidth=1.5)
                          for i in range(len(fold_names))]

# 添加图例 - 调整位置和大小
legend1 = plt.legend(handles=color_legend_elements, loc='upper right',
                     title='Cell Types', title_fontsize=14, framealpha=0.9)
legend2 = plt.legend(handles=marker_legend_elements, loc='lower right',
                     title='Folds', title_fontsize=14, framealpha=0.9)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# 添加标签和标题 - 增加字体大小
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
           fontsize=16, fontweight='bold', labelpad=15)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
           fontsize=16, fontweight='bold', labelpad=15)
plt.title('5-Fold Cross Validation Centroids Visualization (PCA)',
          fontsize=18, fontweight='bold', pad=20)

# 调整坐标轴刻度字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 添加原点参考线
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7)

# 调整布局
plt.tight_layout()

# ============== 导出二维坐标数据 ==============
pca_result_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'CellType': cell_types,
    'Fold': [f'Fold{i+1}' for i in fold_labels]
})
pca_result_df.to_csv('pca_2d_centroids.csv', index=False)
print("已生成 pca_2d_centroids.csv，可导入 Excel/GraphPad 自行绘图。")

# ============== 导出 PCA 载荷（可选） ==============
loads = pca.components_.T * np.sqrt(pca.explained_variance_)
load_df = pd.DataFrame(loads, columns=['PC1', 'PC2'],
                       index=['miR-141', 'miR-155', 'miR-21', 'miR-221', 'miR-222'])
load_df.to_csv('pca_loadings.csv', index=True)
print("已生成 pca_loadings.csv（载荷），可用于 biplot。")

# ========== 载荷稳定性 + 重要特征 ==========
feature_names = load_df.index.to_list()
fold_loadings = []
for fold in fold_names:
    fold_X = np.array([data[fold][ct] for ct in cell_types_list])
    fold_pca = PCA(n_components=2).fit(fold_X)
    fold_loadings.append(fold_pca.components_.T * np.sqrt(fold_pca.explained_variance_))

loading_correlations = np.array(
    [[(np.corrcoef(fold_loadings[i][:, 0], fold_loadings[j][:, 0])[0, 1] +
       np.corrcoef(fold_loadings[i][:, 1], fold_loadings[j][:, 1])[0, 1]) / 2
      for j in range(5)] for i in range(5)]
)
loading_correlations = (loading_correlations + loading_correlations.T) / 2
print("\n各折间PCA载荷平均相关系数:")
print(pd.DataFrame(loading_correlations,
                  index=fold_names,
                  columns=fold_names).round(3))

print("\n=== 最重要的miRNA特征 ===")
for pc in ['PC1', 'PC2']:
    top3 = load_df[pc].abs().sort_values(ascending=False).head(3)
    print(f"{pc}最重要的特征:")
    for rank, (feat, val) in enumerate(top3.items(), 1):
        print(f"  {rank}. {feat} (载荷: {val:.3f})")

# ========== 画载荷图 ==========
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loads[i, 0], loads[i, 1],
              color='r', alpha=0.7, head_width=0.05)
    plt.text(loads[i, 0] * 1.15, loads[i, 1] * 1.15,
             feature, color='black', ha='center', va='center', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Loadings Plot')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

# 显示图形
plt.show()

# 可选：保存高质量图片
# plt.savefig('pca_centroids_visualization.png', dpi=300, bbox_inches='tight')
# plt.savefig('pca_centroids_visualization.pdf', bbox_inches='tight')