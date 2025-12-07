import pandas as pd
import os

def excel_to_csv_multiple_sheets(excel_file):
    try:
        # 读取Excel文件的所有sheet
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        print(f"发现 {len(excel_data)} 个sheet:")
        
        # 遍历每个sheet
        for sheet_name, df in excel_data.items():
            # 清理sheet名字，移除可能导致文件名问题的字符
            clean_sheet_name = sheet_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            
            # 生成CSV文件名
            csv_file = f"{clean_sheet_name}.csv"
            
            # 保存为CSV文件
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            print(f"  - Sheet '{sheet_name}' -> {csv_file}")
            print(f"    数据形状: {df.shape}")
            print(f"    列名: {list(df.columns)}")
            print()
            
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

if __name__ == "__main__":
    excel_file = "机器学习分类数据样本-TXY.xlsx"
    
    if os.path.exists(excel_file):
        excel_to_csv_multiple_sheets(excel_file)
    else:
        print(f"文件 {excel_file} 不存在")