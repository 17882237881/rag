# -*- coding: utf-8 -*-

import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import pdfdeal
from pdfdeal import Doc2X
from pdfdeal.file_tools import get_files, unzips, auto_split_mds


# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# 获取脚本所在的目录，用于构建绝对路径
_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARSED_RESULTS_DIR = os.path.abspath(os.path.join(_SERVICE_DIR, '..', 'parsed_results'))

# 确保目标目录存在
os.makedirs(_PARSED_RESULTS_DIR, exist_ok=True)

def _parse_single_pdf(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    处理单个PDF文件：转换到Markdown、拆分段落、转换图片为在线URL，并保存。
    """
    try:
        # Step 1: 转换PDF到Markdown
        Client = Doc2X(apikey="sk-skwtzc3117q0lp81dc2jena23bhdv3sr")
        out_type = "md"
        file_list, rename_list = get_files(path=os.path.dirname(pdf_path), mode="pdf", out=out_type)
        success, failed, flag = Client.pdf2file(
            pdf_file=[pdf_path],
            output_path=_PARSED_RESULTS_DIR,
            output_names=[os.path.splitext(os.path.basename(pdf_path))[0] + '.md'],
            output_format=out_type,
        )
        if failed:
            return None, str(failed)

        # 处理ZIP文件（如果有）
        zips = [f for f in success if f.endswith(".zip")]
        success, failed, flag = unzips(zip_paths=zips)
        if failed:
            return None, str(failed)

        md_path = os.path.join(_PARSED_RESULTS_DIR, os.path.splitext(os.path.basename(pdf_path))[0] + '.md')

        # Step 2: 拆分段落
        success, failed, flag = auto_split_mds(mdpath=md_path, out_type="replace")
        if failed:
            return None, str(failed)

        # Step 3: 跳过上传，图片已保存到本地

        logging.info(f"文件 {pdf_path} 处理成功，并保存至 {md_path}")
        return md_path, None
    except Exception as e:
        error_message = f"处理PDF文件 {pdf_path} 时发生错误: {e}"
        logging.error(error_message, exc_info=True)
        return None, error_message

def parse_pdfs_to_markdown(pdf_paths: List[str], max_workers: int = 4) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    并发地将多个PDF文件转换为增强的Markdown文本并保存到文件。
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_parse_single_pdf, pdf_paths))
    
    return results

# --- 示例用法 (Example Usage) ---
if __name__ == "__main__":
    example_pdf_path = "C:\\Users\\liujilong\\Desktop\\8131d40002d48f0b4dfc3cfc44f7da75.pdf"
    if os.path.exists(example_pdf_path):
        example_paths = [example_pdf_path]

        logging.info(f"开始批量处理 {len(example_paths)} 个PDF文件...")
        
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
