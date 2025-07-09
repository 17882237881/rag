# -*- coding: utf-8 -*-

import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 获取脚本所在的目录，用于构建绝对路径
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARSED_RESULTS_DIR = os.path.abspath(os.path.join(_SERVICE_DIR, '..', 'parsed_results'))

# 确保目标目录存在
os.makedirs(_PARSED_RESULTS_DIR, exist_ok=True)

def _parse_single_excel(excel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    处理单个Excel文件，将其所有工作表转换为Markdown，并保存到文件。
    Internal function to process a single Excel file, convert all its sheets to Markdown, and save to a file.
    """
    if not os.path.exists(excel_path):
        error_message = f"错误：在路径 {excel_path} 未找到Excel文件。"
        logging.error(error_message)
        return None, error_message

    try:
        base_name = os.path.basename(excel_path)
        file_name, _ = os.path.splitext(base_name)
        output_filename = f"{file_name}.md"
        output_path = os.path.join(_PARSED_RESULTS_DIR, output_filename)

        logging.info(f"开始转换Excel文件: {excel_path}")
        
        # 读取所有工作表
        # sheet_name=None 会返回一个字典，键是工作表名称，值是DataFrame
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        
        all_sheets_markdown = []
        for sheet_name, df in excel_data.items():
            # 添加工作表标题
            all_sheets_markdown.append(f"## 工作表: {sheet_name}\n")
            # 将DataFrame转换为Markdown表格，不包含索引
            all_sheets_markdown.append(df.to_markdown(index=False))
            all_sheets_markdown.append("\n\n") # 添加空行以便阅读
        
        combined_markdown = "".join(all_sheets_markdown)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_markdown)
            
        logging.info(f"文件 {excel_path} 转换成功，并保存至 {output_path}")
        return output_path, None
    except Exception as e:
        error_message = f"转换Excel文件 {excel_path} 时发生未知错误: {e}"
        logging.error(error_message, exc_info=True)
        return None, error_message

def parse_excel_to_markdown(excel_paths: List[str], max_workers: int = 4) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    并发地将多个Excel文件转换为Markdown文本并保存到文件。
    Concurrently converts multiple Excel files to Markdown text and saves them to files.

    Args:
        excel_paths (List[str]): 一个包含多个Excel文件路径的列表。
                               A list of paths to Excel files.
        max_workers (int): 用于并发处理的线程池的最大线程数。
                           The maximum number of threads to use for concurrent processing.

    Returns:
        一个结果元组的列表。每个元组包含：
        - str: 成功时输出文件的路径。
        - str: 失败时的错误信息。
        A list of result tuples. Each tuple contains:
        - str: The path to the output file on success.
        - str: An error message on failure.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parse_single_excel, excel_paths))
    
    return results

# --- 示例用法 (Example Usage) ---
if __name__ == "__main__":
    # 重要：请将此路径替换为您自己的Excel文件的实际路径
    # example_excel_path = "/path/to/your/document.xlsx"
    
    # 由于我无法访问您的文件系统，请在此处提供一个有效的Excel文件路径以进行测试
    # Since I cannot access your filesystem, please provide a valid Excel file path for testing.
    # 请确保您的环境中安装了pandas和openpyxl (或xlrd)库：pip install pandas openpyxl
    example_excel_path = "/Users/liujilong/Desktop/工作文档/结算完整性审查知识库/结算完整性审查.xlsx" # 替换为您的实际Excel文件路径

    if example_excel_path and os.path.exists(example_excel_path):
        example_paths = [
            example_excel_path
        ]

        logging.info(f"开始批量处理 {len(example_paths)} 个Excel文件...")
        
        # 调用批量处理函数
        all_results = parse_excel_to_markdown(example_paths)

        logging.info("--- 批量处理结果 ---")
        for i, (output_path, error) in enumerate(all_results):
            logging.info(f"文件 {example_paths[i]}:")
            if error:
                logging.error(f"  处理失败: {error}")
            else:
                logging.info(f"  处理成功！结果已保存至: {output_path}")
        logging.info("--- 处理完成 ---")
    else:
        logging.warning(f"未找到示例文件 '{example_excel_path}'，无法运行示例。请确保文件存在且路径正确。")
