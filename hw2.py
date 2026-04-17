import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

result_dir = Path("results/hw2")
result_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/BeijingHotel.csv")

# 数据预处理
df["log_房价"] = np.log(df["房价"])
df["装修距今年数"] = 2026 - df["装修时间"]

# EDA
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["房价"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("房价分布 (原始数据)")
sns.histplot(df["log_房价"], kde=True, ax=axes[1], color="lightgreen")
axes[1].set_title("房价分布 (Log变换后)")
plt.tight_layout()
plt.savefig(result_dir / "01_price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(
    corr_matrix[["log_房价", "房价"]].sort_values(by="log_房价", ascending=False),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
plt.title("各特征与房价的相关系数")
plt.tight_layout()
plt.savefig(result_dir / "02_correlation_heatmap.png", dpi=300)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
sns.boxplot(x="地区", y="log_房价", data=df, ax=axes[0], palette="Set2")
axes[0].set_title("不同地区的房价(Log)分布")
axes[0].tick_params(axis="x", rotation=45)
sns.boxplot(x="房间类型", y="log_房价", data=df, ax=axes[1], palette="Set3")
axes[1].set_title("不同房间类型的房价(Log)分布")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(result_dir / "03_categorical_boxplots.png", dpi=300)
plt.close()

# Clustering and ANOVA
# 地理 K-Means 聚类
coords = df[["经度", "纬度"]]
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["地理商圈"] = kmeans.fit_predict(coords_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(
    x="经度", y="纬度", hue="地理商圈", palette="Set1", data=df, ax=axes[0], s=60
)
axes[0].set_title("北京高端酒店地理聚类 (空间分布)")
sns.boxplot(x="地理商圈", y="log_房价", data=df, palette="Set1", ax=axes[1])
axes[1].set_title("不同地理商圈的房价分布 (Log变换后)")
plt.tight_layout()
plt.savefig(result_dir / "04_kmeans_clusters.png", dpi=300)
plt.close()

# ANOVA 检验
model_area = ols("log_房价 ~ C(地区)", data=df).fit()
anova_area = sm.stats.anova_lm(model_area, typ=2)

model_room = ols("log_房价 ~ C(房间类型)", data=df).fit()
anova_room = sm.stats.anova_lm(model_room, typ=2)

with open(result_dir / "05_anova_results.txt", "w", encoding="utf-8") as f:
    f.write("=== 地区对房价(Log)的方差分析 ===\n")
    f.write(anova_area.to_string())
    f.write("\n\n=== 房间类型对房价(Log)的方差分析 ===\n")
    f.write(anova_room.to_string())


# OLS
formula = (
    "log_房价 ~ 卫生评分 + 服务评分 + 设施评分 + 位置评分 + 评价数 + "
    "装修距今年数 + 公司 + 出行住宿 + 校园生活 + "
    "C(地区) + C(房间类型) + C(地理商圈)"
)

final_model = ols(formula, data=df).fit()

with open(result_dir / "06_regression_summary.txt", "w", encoding="utf-8") as f:
    f.write(final_model.summary().as_text())

df.to_csv(result_dir / "processed_BeijingHotel.csv", index=False, encoding="utf-8-sig")
