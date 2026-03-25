import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def setup_environment():
    """设置绘图风格、中文显示并创建输出目录"""
    # 中文显示配置
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="SimHei")

    # 创建保存目录
    output_dir = "results/hw1/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    return output_dir


def plot_scoring_vs_salary(df, save_path):
    """进攻能力与薪金的关系（散点图）"""
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        x="进攻能力",
        y="球员薪金",
        hue="是否入选过全明星",
        size="场均时间",
        sizes=(20, 200),
        data=df,
        alpha=0.7,
        palette="magma",
    )
    plt.title("进攻能力 vs 球员薪金 (气泡大小代表场均时间)", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "02_scoring_vs_salary.png"), dpi=300)
    plt.show()


def plot_physique_by_position(df, save_path):
    """按位置分析球员平均身高"""
    plt.figure(figsize=(10, 6))
    order = df.groupby("位置")["身高"].mean().sort_values().index

    sns.barplot(
        x="位置",
        y="身高",
        data=df,
        palette="coolwarm",
        order=order,
        hue="位置",
        legend=False,
    )

    plt.title("NBA 各位置平均身高对比", fontsize=15)
    plt.xlabel("场上位置")
    plt.ylabel("身高 (米)")
    plt.ylim(df["身高"].min() - 0.05, df["身高"].max() + 0.05)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "01_physique_analysis.png"), dpi=300)
    plt.show()


def plot_allstar_comparison(df, save_path):
    """全明星与普通球员的薪金分布（箱线图）"""
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x="是否入选过全明星",
        y="球员薪金",
        data=df,
        palette="Set2",
        hue="是否入选过全明星",
        legend=False,
    )

    sns.stripplot(
        x="是否入选过全明星", y="球员薪金", data=df, color="orange", alpha=0.3
    )

    plt.title("全明星 vs 普通球员薪金分布", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "03_allstar_salary_dist.png"), dpi=300)
    plt.show()


def plot_team_impact_heatmap(df, save_path):
    """球队背景对薪金的影响（热力图）"""
    pivot = df.pivot_table(
        values="球员薪金", index="球队胜率", columns="球队市值", aggfunc="mean"
    )
    order = ["高", "中", "低"]
    pivot = pivot.reindex(
        index=[x for x in order if x in pivot.index],
        columns=[x for x in order if x in pivot.columns],
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar_kws={"label": "平均薪金 (万美元)"},
    )
    plt.title("球队胜率与市值交叉下的平均薪金分布", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "04_team_impact_heatmap.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    # 路径配置
    data_path = "data/NBAplayers.csv"

    # 1. 环境准备
    output_path = setup_environment()

    # 2. 读取数据 (包含之前解决的编码问题)
    try:
        df_nba = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_nba = pd.read_csv(data_path, encoding="gbk")

    # 3. 执行绘图并保存
    print("正在处理并保存图表到 results/hw1/ ...")

    plot_physique_by_position(df_nba, output_path)
    plot_scoring_vs_salary(df_nba, output_path)
    plot_allstar_comparison(df_nba, output_path)
    plot_team_impact_heatmap(df_nba, output_path)

    print(f"\n所有统计图表已成功保存至: {os.path.abspath(output_path)}")
