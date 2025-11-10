#!/usr/bin/env python3
"""查看和分析 identities.parquet 文件的内容"""

import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 定义文件路径
    identities_path = Path("outputs/embeddings/identities.parquet")
    
    # 检查文件是否存在
    if not identities_path.exists():
        print(f"错误: 文件 {identities_path} 不存在")
        print("请先运行 cluster_identities.py 脚本生成该文件")
        sys.exit(1)
    
    # 读取数据
    df = pd.read_parquet(identities_path)
    
    # 创建可视化图表
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('海狗面部识别聚类结果分析', fontsize=16, fontweight='bold')
    
    # 1. 身份分布柱状图
    ax1 = plt.subplot(2, 3, 1)
    identity_counts = df['identity'].value_counts().sort_index()
    colors = ['red' if x == -1 else 'skyblue' for x in identity_counts.index]
    bars = plt.bar(range(len(identity_counts)), identity_counts.values, color=colors)
    plt.xlabel('身份ID')
    plt.ylabel('样本数量')
    plt.title('各身份样本分布')
    plt.xticks(range(len(identity_counts)), [f'{x}' if x != -1 else '未分类(-1)' for x in identity_counts.index], rotation=45)
    
    # 在柱状图上添加数值标签
    for i, (idx, count) in enumerate(identity_counts.items()):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    # 2. 身份分布饼图
    ax2 = plt.subplot(2, 3, 2)
    # 排除未分类(-1)的数据制作饼图
    valid_identity_counts = identity_counts[identity_counts.index != -1]
    if len(valid_identity_counts) > 0:
        plt.pie(valid_identity_counts.values, labels=[f'身份 {x}' for x in valid_identity_counts.index], autopct='%1.1f%%')
        plt.title('各身份占比分布')
    else:
        plt.text(0.5, 0.5, '无有效分类数据', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    
    # 3. 置信度分布直方图
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(df['score'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.xlabel('检测置信度')
    plt.ylabel('频次')
    plt.title('面部检测置信度分布')
    plt.axvline(df['score'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'平均值: {df["score"].mean():.3f}')
    plt.legend()
    
    # 4. 每个身份的置信度箱线图
    ax4 = plt.subplot(2, 3, 4)
    # 只显示有效身份（排除-1）
    valid_data = df[df['identity'] != -1].copy()
    if len(valid_data) > 0:
        valid_data['identity_str'] = valid_data['identity'].apply(lambda x: f'身份 {x}')
        sns.boxplot(data=valid_data, x='identity_str', y='score', ax=ax4)
        plt.xlabel('身份')
        plt.ylabel('检测置信度')
        plt.title('各身份置信度分布箱线图')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, '无有效分类数据', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    
    # 5. 图像来源统计
    ax5 = plt.subplot(2, 3, 5)
    image_counts = df['image_path'].value_counts()
    if len(image_counts) > 0:
        plt.hist(image_counts, bins=min(20, len(image_counts)), color='orange', edgecolor='black', alpha=0.7)
        plt.xlabel('每张图片中的面部数量')
        plt.ylabel('图片数量')
        plt.title('每张图片面部数量分布')
        plt.axvline(image_counts.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'平均值: {image_counts.mean():.1f}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
    
    # 6. 数据汇总表
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 准备统计数据
    stats_data = [
        ['总记录数', str(len(df))],
        ['不同身份总数', str(df['identity'].nunique())],
        ['有效身份数', str((df['identity'] != -1).sum())],
        ['未分类面部数', str((df['identity'] == -1).sum())],
        ['平均置信度', f"{df['score'].mean():.3f}"],
        ['最高置信度', f"{df['score'].max():.3f}"],
        ['最低置信度', f"{df['score'].min():.3f}"],
        ['不同图像数', str(df['image_path'].nunique())]
    ]
    
    table = ax6.table(cellText=stats_data,
                      colLabels=['统计项', '数值'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('数据统计汇总')
    
    plt.tight_layout()
    plt.savefig('outputs/embeddings/identity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("分析完成！图表已保存至: outputs/embeddings/identity_analysis.png")

if __name__ == "__main__":
    main()