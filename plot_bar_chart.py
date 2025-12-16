import matplotlib.pyplot as plt
import numpy as np

def plot_nme_comparison():
    # 数据
    models = ['MSE Loss (Baseline)', 'Wing Loss (Ours)']
    nme_values = [0.0343, 0.0311] # 你的真实数据
    improvement = 9.34 # 你的提升百分比

    # 设置风格
    plt.figure(figsize=(8, 6))
    # 颜色：灰色代表基准，红色代表你的改进
    colors = ['#bdc3c7', '#e74c3c'] 
    
    # 画柱状图
    bars = plt.bar(models, nme_values, color=colors, width=0.5)
    
    # 设置Y轴范围，留出一点头部空间画箭头
    plt.ylim(0, 0.045)
    plt.ylabel('Normalized Mean Error (NME)', fontsize=12)
    plt.title('Performance Comparison on Test Set', fontsize=14, fontweight='bold')
    
    # 在柱子上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.0005,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 画提升箭头
    # 计算两个柱子的顶端坐标
    x1 = bars[0].get_x() + bars[0].get_width()/2.0
    y1 = bars[0].get_height()
    x2 = bars[1].get_x() + bars[1].get_width()/2.0
    y2 = bars[1].get_height()

    # 画一条线连接两边
    plt.annotate('', xy=(x2, y2 + 0.002), xytext=(x1, y2 + 0.002),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # 写上提升百分比
    mid_x = (x1 + x2) / 2
    plt.text(mid_x, y2 + 0.003, f'Improvement: {improvement}%', 
             ha='center', va='bottom', fontsize=11, color='green', fontweight='bold')

    # 去掉顶部和右边的边框，看起来更学术
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('nme_comparison_bar.png', dpi=300)
    print("柱状图已保存为 nme_comparison_bar.png")
    plt.show()

if __name__ == "__main__":
    plot_nme_comparison()