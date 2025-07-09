# -*- coding: utf-8 -*-

import os
import logging
import pypandoc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 获取脚本所在的目录，用于构建绝对路径
# Get the directory where the script is located to build absolute paths
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARSED_RESULTS_DIR = os.path.abspath(os.path.join(_SERVICE_DIR, '..', 'parsed_results'))

# 确保目标目录存在
# Ensure the target directory exists
os.makedirs(_PARSED_RESULTS_DIR, exist_ok=True)

def _parse_single_docx(docx_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    使用 pandoc 处理单个DOCX文件，并将结果保存到文件中。
    Internal function to process a single DOCX file using pandoc and save the result to a file.
    """
    if not os.path.exists(docx_path):
        error_message = f"错误：在路径 {docx_path} 未找到DOCX文件。"
        logging.error(error_message)
        return None, error_message

    try:
        base_name = os.path.basename(docx_path)
        file_name, _ = os.path.splitext(base_name)
        output_filename = f"{file_name}.md"
        output_path = os.path.join(_PARSED_RESULTS_DIR, output_filename)

        logging.info(f"开始转换DOCX文件: {docx_path}")
        # 调用 pypandoc 进行转换，指定输出格式为 markdown
        # Call pypandoc to convert, specifying the output format as markdown
        pypandoc.convert_file(docx_path, 'markdown', outputfile=output_path)
        logging.info(f"文件 {docx_path} 转换成功，并保存至 {output_path}")
        return output_path, None
    except Exception as e:
        error_message = f"转换DOCX文件 {docx_path} 时发生未知错误: {e}"
        logging.error(error_message, exc_info=True)
        return None, error_message

def parse_docx_to_markdown(docx_paths: List[str], max_workers: int = 4) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    并发地将多个DOCX文件转换为Markdown，并将结果保存到文件。
    Concurrently converts multiple DOCX files to Markdown and saves the results to files.

    Args:
        docx_paths (List[str]): 一个包含多个DOCX文件路径的列表。
                               A list of paths to DOCX files.
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
        results = list(executor.map(_parse_single_docx, docx_paths))
    
    return results

# --- 示例用法 (Example Usage) ---
if __name__ == "__main__":
    # 重要：请将此路径替换为您自己的DOCX文件的实际路径
    # example_docx_path = "/path/to/your/document.docx"
    
    # 由于我无法访问您的文件系统，请在此处提供一个有效的DOCX文件路径以进行测试
    # Since I cannot access your filesystem, please provide a valid DOCX file path for testing.
    example_docx_path = "/Users/liujilong/Desktop/工作文档/银河证券四期-知识库问答材料-2025.06.25 3/电子章需求说明书_V终版.docx"

    if example_docx_path and os.path.exists(example_docx_path):
        example_paths = [example_docx_path, "/path/to/non_existent_file.docx"]

        logging.info(f"开始批量处理 {len(example_paths)} 个DOCX文件...")
        
        all_results = parse_docx_to_markdown(example_paths)

        logging.info("--- 批量处理结果 ---")
        for i, (output_path, error) in enumerate(all_results):
            logging.info(f"文件 {example_paths[i]}:")
            if error:
                logging.error(f"  处理失败: {error}")
            else:
                logging.info(f"  处理成功！结果已保存至: {output_path}")
        logging.info("--- 处理完成 ---")
    else:
        logging.warning("请在脚本中设置 'example_docx_path' 为一个有效的DOCX文件路径以运行示例。")

