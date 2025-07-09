# -*- coding: utf-8 -*-

import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 获取脚本所在的目录，用于构建绝对路径
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARSED_RESULTS_DIR = os.path.abspath(os.path.join(_SERVICE_DIR, '..', 'parsed_results'))

# 确保目标目录存在
os.makedirs(_PARSED_RESULTS_DIR, exist_ok=True)

# 模块级变量，用于缓存模型和确保线程安全的锁
_converter: Optional[PdfConverter] = None
_model_lock = threading.Lock()

def _initialize_and_get_converter() -> PdfConverter:
    """
    内部函数，使用双重检查锁定模式确保线程安全地初始化模型。
    Internal function to thread-safely initialize the model using a double-checked locking pattern.
    """
    global _converter
    if _converter is None:
        with _model_lock:
            if _converter is None:
                logging.info("首次调用，正在加载PDF解析模型... (可能需要一些时间)")
                try:
                    _converter = PdfConverter(artifact_dict=create_model_dict())
                    logging.info("PDF解析模型加载成功。")
                except Exception as e:
                    logging.error(f"加载PDF解析模型时发生严重错误: {e}", exc_info=True)
                    raise RuntimeError(f"无法初始化PDF解析模型: {e}") from e
    return _converter

def _parse_single_pdf(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    处理单个PDF文件并将其内容保存到Markdown文件。
    Internal function to process a single PDF file and save its content to a Markdown file.
    """
    try:
        converter = _initialize_and_get_converter()
    except RuntimeError as e:
        return None, str(e)

    if not os.path.exists(pdf_path):
        error_message = f"错误：在路径 {pdf_path} 未找到PDF文件。"
        logging.error(error_message)
        return None, error_message

    try:
        base_name = os.path.basename(pdf_path)
        file_name, _ = os.path.splitext(base_name)
        output_filename = f"{file_name}.md"
        output_path = os.path.join(_PARSED_RESULTS_DIR, output_filename)

        logging.info(f"开始转换PDF文件: {pdf_path}")
        rendered = converter(pdf_path)
        markdown_text, _, _ = text_from_rendered(rendered)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
            
        logging.info(f"文件 {pdf_path} 转换成功，并保存至 {output_path}")
        return output_path, None
    except Exception as e:
        error_message = f"转换PDF文件 {pdf_path} 时发生未知错误: {e}"
        logging.error(error_message, exc_info=True)
        return None, error_message

def parse_pdfs_to_markdown(pdf_paths: List[str], max_workers: int = 4) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    并发地将多个PDF文件转换为Markdown文本并保存到文件。
    Concurrently converts multiple PDF files to Markdown text and saves them to files.

    Args:
        pdf_paths (List[str]): 一个包含多个PDF文件路径的列表。
                               A list of paths to PDF files.
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
    # 确保模型在进入线程池前至少被尝试加载一次，简化线程逻辑
    try:
        _initialize_and_get_converter()
    except RuntimeError as e:
        # 如果模型加载失败，所有文件都无法处理，直接返回错误
        error_result = (None, str(e))
        return [error_result] * len(pdf_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parse_single_pdf, pdf_paths))
    
    return results

# --- 示例用法 (Example Usage) ---
if __name__ == "__main__":
    # 创建一个包含多个PDF文件路径的列表（在这里我们使用同一个文件多次作为示例）
    # Create a list of PDF paths (here we use the same file multiple times for demonstration)
    example_pdf_path = "/Users/liujilong/Desktop/2310.12931v2.pdf"
    if os.path.exists(example_pdf_path):
        example_paths = [
            example_pdf_path
        ]

        logging.info(f"开始批量处理 {len(example_paths)} 个PDF文件...")
        
        # 调用批量处理函数
        all_results = parse_pdfs_to_markdown(example_paths)

        logging.info("--- 批量处理结果 ---")
        for i, (output_path, error) in enumerate(all_results):
            logging.info(f"文件 {example_paths[i]}:")
            if error:
                logging.error(f"  处理失败: {error}")
            else:
                logging.info(f"  处理成功！结果已保存至: {output_path}")
        logging.info("--- 处理完成 ---")
    else:
        logging.warning(f"未找到示例文件 '{example_pdf_path}'，无法运行示例。")
