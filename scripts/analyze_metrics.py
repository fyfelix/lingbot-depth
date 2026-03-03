import os
import sys
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Evaluation Metrics")
    parser.add_argument("--csv", type=str, required=True, help="Path to the all_metrics_*.csv file")
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = args.csv
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("CSV 文件分析报告")
    print("=" * 80)
    print(f"\n总记录数: {len(df)}")
    print(f"数据形状: {df.shape}")

    # 越低越好的指标（升序排序）
    lower_is_better = ['L1', 'rmse_linear', 'abs_relative_difference']

    # 越高越好的指标（降序排序）
    higher_is_better = ['delta4_acc_105', 'delta5_acc110', 'delta1_acc']

    print("\n" + "=" * 80)
    print("【越低越好】指标排序 - 表现最差的样本（后10名）")
    print("=" * 80)

    for metric in lower_is_better:
        if metric not in df.columns: continue
        print(f"\n【{metric}】- 表现最差的10个样本:")
        sorted_df = df.sort_values(by=metric, ascending=True)
        worst_samples = sorted_df.tail(10)
        print(f"最差值: {worst_samples[metric].iloc[-1]:.6f}")
        for idx, row in worst_samples.iterrows():
            print(f"  {row['name']} : {row[metric]:.6f}")

    print("\n" + "=" * 80)
    print("【越高越好】指标排序 - 表现最差的样本（后10名）")
    print("=" * 80)

    for metric in higher_is_better:
        if metric not in df.columns: continue
        print(f"\n【{metric}】- 表现最差的10个样本:")
        sorted_df = df.sort_values(by=metric, ascending=False)
        worst_samples = sorted_df.tail(10)
        print(f"最差值: {worst_samples[metric].iloc[-1]:.6f}")
        for idx, row in worst_samples.iterrows():
            print(f"  {row['name']} : {row[metric]:.6f}")

    print("\n" + "=" * 80)
    print("综合统计分析")
    print("=" * 80)
    
    available_metrics = [m for m in lower_is_better + higher_is_better if m in df.columns]
    print("\n所有指标的统计信息:")
    print(df[available_metrics].describe().T)

if __name__ == "__main__":
    main()
