from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
import pdfplumber
import fitz  # PyMuPDF
import logging
import os
from datetime import datetime
import json
import re
from typing import List, Dict, Tuple
from markdownify import markdownify

logger = logging.getLogger(__name__)
"""
PDF文档加载服务类
    这个服务类提供了多种PDF文档加载方法，支持不同的加载策略和分块选项。
    主要功能：
    1. 支持多种PDF解析库：
        - PyMuPDF (fitz): 适合快速处理大量PDF文件，性能最佳
        - PyPDF: 适合简单的PDF文本提取，依赖较少
        - pdfplumber: 适合需要处理表格或需要文本位置信息的场景
        - unstructured: 适合需要更好的文档结构识别和灵活分块策略的场景
    
    2. 文档加载特性：
        - 保持页码信息
        - 支持文本分块
        - 提供元数据存储
        - 支持不同的加载策略（使用unstructured时）
 """
class LoadingService:
    """
    PDF文档加载服务类，提供多种PDF文档加载和处理方法。
    
    属性:
        total_pages (int): 当前加载PDF文档的总页数
        current_page_map (list): 存储当前文档的页面映射信息，每个元素包含页面文本和页码
    """
    
    def __init__(self):
        self.total_pages = 0
        self.current_page_map = []
    
    def load_pdf(self, file_path: str, method: str, strategy: str = None, chunking_strategy: str = None, chunking_options: dict = None, output_format: str = "markdown") -> str:
        """
        加载PDF文档的主方法，支持多种加载策略。

        参数:
            file_path (str): PDF文件路径
            method (str): 加载方法，支持 'pymupdf', 'pypdf', 'pdfplumber', 'unstructured'
            strategy (str, optional): 使用unstructured方法时的策略，可选 'fast', 'hi_res', 'ocr_only'
            chunking_strategy (str, optional): 文本分块策略，可选 'basic', 'by_title'
            chunking_options (dict, optional): 分块选项配置
            output_format (str, optional): 输出格式，可选 'text', 'markdown'

        返回:
            str: 提取的文本内容（根据output_format格式化）
        """
        try:
            # 首先使用指定方法加载PDF内容
            if method == "pymupdf":
                text_content = self._load_with_pymupdf(file_path)
            elif method == "pypdf":
                text_content = self._load_with_pypdf(file_path)
            elif method == "pdfplumber":
                text_content = self._load_with_pdfplumber(file_path)
            elif method == "unstructured":
                text_content = self._load_with_unstructured(
                    file_path, 
                    strategy=strategy,
                    chunking_strategy=chunking_strategy,
                    chunking_options=chunking_options
                )
            else:
                raise ValueError(f"Unsupported loading method: {method}")
            
            # 根据输出格式处理内容
            if output_format == "markdown":
                return self._convert_to_markdown()
            else:
                return text_content
                
        except Exception as e:
            logger.error(f"Error loading PDF with {method}: {str(e)}")
            raise
    
    def get_total_pages(self) -> int:
        """
        获取当前加载文档的总页数。

        返回:
            int: 文档总页数
        """
        return max(page_data['page'] for page_data in self.current_page_map) if self.current_page_map else 0
    
    def get_page_map(self) -> list:
        """
        获取当前文档的页面映射信息。

        返回:
            list: 包含每页文本内容和页码的列表
        """
        return self.current_page_map
    
    def _load_with_pymupdf(self, file_path: str) -> str:
        """
        使用PyMuPDF库加载PDF文档，提取丰富的文档结构信息。
        适合快速处理大量PDF文件，性能最佳。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with fitz.open(file_path) as doc:
                self.total_pages = len(doc)  # 存储总页数
                for page_num, page in enumerate(doc, 1): # 从1开始计数
                    # 提取结构化文本块信息
                    page_blocks = self._extract_structured_blocks(page, page_num)
                    if page_blocks:
                        text_blocks.extend(page_blocks)
                        
            self.current_page_map = text_blocks  # 存储页面映射
            return "\n".join(block["text"] for block in text_blocks) 
        except Exception as e:
            logger.error(f"PyMuPDF error: {str(e)}")
            raise
    
    def _extract_structured_blocks(self, page, page_num: int) -> List[Dict]:
        """
        从PDF页面提取结构化文本块，包括字体、样式、位置等信息。
        
        参数:
            page: PyMuPDF页面对象
            page_num (int): 页码
            
        返回:
            List[Dict]: 结构化文本块列表
        """
        blocks = []
        
        try:
            # 获取文本块信息，包括字体和位置
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:  # 跳过图像块
                    continue
                    
                block_text = ""
                block_info = {
                    "bbox": block.get("bbox", [0, 0, 0, 0]),  # 边界框
                    "fonts": [],  # 字体信息
                    "font_sizes": [],  # 字体大小
                    "font_flags": [],  # 字体标志（粗体、斜体等）
                    "is_title": False,  # 是否为标题
                    "is_list": False,  # 是否为列表
                    "is_table": False,  # 是否为表格
                }
                
                for line in block["lines"]:
                    line_text = ""
                    line_fonts = []
                    line_sizes = []
                    line_flags = []
                    
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            line_text += span_text
                            line_fonts.append(span.get("font", ""))
                            line_sizes.append(span.get("size", 0))
                            line_flags.append(span.get("flags", 0))
                    
                    if line_text.strip():
                        block_text += line_text + "\n"
                        block_info["fonts"].extend(line_fonts)
                        block_info["font_sizes"].extend(line_sizes)
                        block_info["font_flags"].extend(line_flags)
                
                if block_text.strip():
                    # 分析文本块特征
                    block_info = self._analyze_block_features(block_text.strip(), block_info)
                    
                    blocks.append({
                        "text": block_text.strip(),
                        "page": page_num,
                        "structure_info": block_info
                    })
                    
        except Exception as e:
            logger.warning(f"Error extracting structured blocks from page {page_num}: {str(e)}")
            # 降级到简单文本提取
            simple_text = page.get_text("text")
            if simple_text.strip():
                blocks.append({
                    "text": simple_text.strip(),
                    "page": page_num,
                    "structure_info": {"is_simple": True}
                })
        
        return blocks
    
    def _analyze_block_features(self, text: str, block_info: Dict) -> Dict:
        """
        分析文本块特征，判断是否为标题、列表、表格等。
        
        参数:
            text (str): 文本内容
            block_info (Dict): 块信息
            
        返回:
            Dict: 更新后的块信息
        """
        # 分析字体大小
        avg_font_size = sum(block_info["font_sizes"]) / len(block_info["font_sizes"]) if block_info["font_sizes"] else 12
        max_font_size = max(block_info["font_sizes"]) if block_info["font_sizes"] else 12
        
        # 分析字体标志（粗体、斜体等）
        has_bold = any(flag & 2**4 for flag in block_info["font_flags"])  # 粗体标志
        has_italic = any(flag & 2**1 for flag in block_info["font_flags"])  # 斜体标志
        
        # 判断是否为标题
        is_title = (
            len(text) < 100 and  # 标题通常较短
            (avg_font_size > 14 or has_bold) and  # 字体较大或粗体
            not text.endswith('.') and  # 标题通常不以句号结尾
            len(text.split('\n')) <= 3  # 标题通常不超过3行
        )
        
        # 判断标题级别
        title_level = 1
        if avg_font_size > 18:
            title_level = 1
        elif avg_font_size > 16:
            title_level = 2
        elif avg_font_size > 14:
            title_level = 3
        else:
            title_level = 4
            
        # 判断是否为列表
        is_list = bool(re.match(r'^\s*[-•·*]\s+', text) or 
                      re.match(r'^\s*\d+[\.\)]\s+', text) or
                      re.match(r'^\s*[a-zA-Z][\.\)]\s+', text))
        
        # 判断是否为表格
        is_table = bool('\t' in text or 
                       text.count('|') > 2 or
                       re.search(r'\s{4,}', text))  # 多个空格可能表示列对齐
        
        block_info.update({
            "is_title": is_title,
            "title_level": title_level if is_title else 0,
            "is_list": is_list,
            "is_table": is_table,
            "avg_font_size": avg_font_size,
            "max_font_size": max_font_size,
            "has_bold": has_bold,
            "has_italic": has_italic
        })
        
        return block_info
    
    def _load_with_pypdf(self, file_path: str) -> str:
        """
        使用PyPDF库加载PDF文档。
        适合简单的PDF文本提取，依赖较少。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        try:
            text_blocks = []
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"PyPDF error: {str(e)}")
            raise
    
    def _load_with_unstructured(self, file_path: str, strategy: str = "fast", chunking_strategy: str = "basic", chunking_options: dict = None) -> str:
        """
        使用unstructured库加载PDF文档。
        适合需要更好的文档结构识别和灵活分块策略的场景。
        
        unstructured库提供了多种处理策略：
        - fast: 快速处理，性能优先
        - hi_res: 高分辨率处理，质量优先
        - ocr_only: 仅使用OCR识别
        
        分块策略包括：
        - basic: 基于字符数的基础分块
        - by_title: 基于标题的智能分块
    
        参数:
            file_path (str): PDF文件路径
            strategy (str): 加载策略，默认'fast'
                - 'fast': 快速处理模式，适合大批量文档
                - 'hi_res': 高分辨率模式，文档结构识别更准确
                - 'ocr_only': 纯OCR模式，适合扫描文档
            chunking_strategy (str): 分块策略，默认'basic'
                - 'basic': 基于字符数的固定大小分块
                - 'by_title': 基于文档标题结构的智能分块
            chunking_options (dict): 分块选项配置
                - maxCharacters: 最大字符数
                - newAfterNChars: 新块起始字符数
                - combineTextUnderNChars: 合并小块的阈值
                - overlap: 块间重叠字符数
                - overlapAll: 是否对所有块应用重叠
    
        返回:
            str: 提取的文本内容
            
        异常:
            Exception: 当文档处理失败时抛出异常
        """
        try:
            # 定义不同处理策略的参数配置
            # fast: 快速处理，平衡性能和质量
            # hi_res: 高分辨率处理，更好的结构识别但速度较慢
            # ocr_only: 纯OCR模式，适合扫描版PDF
            strategy_params = {
                "fast": {"strategy": "fast"},
                "hi_res": {"strategy": "hi_res"},
                "ocr_only": {"strategy": "ocr_only"}
            }            
         
            # 根据分块策略准备参数
            chunking_params = {}
            
            # 基础分块策略：基于字符数进行固定大小分块
            if chunking_strategy == "basic":
                chunking_params = {
                    # 每个块的最大字符数
                    "max_characters": chunking_options.get("maxCharacters", 4000),
                    # 在多少字符后开始新块
                    "new_after_n_chars": chunking_options.get("newAfterNChars", 3000),
                    # 合并小于此字符数的文本块
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    # 块之间的重叠字符数，用于保持上下文连续性
                    "overlap": chunking_options.get("overlap", 200),
                    # 是否对所有块应用重叠策略
                    "overlap_all": chunking_options.get("overlapAll", False)
                }
            # 按标题分块策略：基于文档结构进行智能分块
            elif chunking_strategy == "by_title":
                chunking_params = {
                    # 指定按标题分块
                    "chunking_strategy": "by_title",
                    # 合并小于此字符数的文本块
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    # 是否允许跨页面的章节
                    "multipage_sections": chunking_options.get("multiPageSections", False)
                }
            
            # 合并策略参数和分块参数
            # 使用字典解包操作符合并两个参数字典
            params = {**strategy_params.get(strategy, {"strategy": "fast"}), **chunking_params}
            
            # 调用unstructured库的partition_pdf函数处理PDF
            # 这是核心处理步骤，返回文档元素列表
            elements = partition_pdf(file_path, **params)
            
            # 调试日志：记录每个元素的详细信息
            # 在生产环境中可以移除或设置为更高的日志级别
            for elem in elements:
                logger.debug(f"Element type: {type(elem)}")  # 元素类型
                logger.debug(f"Element content: {str(elem)}")  # 元素内容
                logger.debug(f"Element dir: {dir(elem)}")  # 元素属性列表
            
            # 初始化文本块列表和页面集合
            text_blocks = []
            pages = set()  # 使用集合自动去重页码
            
            # 遍历所有解析出的文档元素
            for elem in elements:
                # 获取元素的元数据字典
                metadata = elem.metadata.__dict__
                # 提取页码信息
                page_number = metadata.get('page_number')
                
                # 只处理有页码信息的元素
                if page_number is not None:
                    # 记录页码到集合中
                    pages.add(page_number)
                    
                    # 清理元数据，确保所有值都可以JSON序列化
                    # 这对后续的数据存储和传输很重要
                    cleaned_metadata = {}
                    for key, value in metadata.items():
                        # 跳过内部字段
                        if key == '_known_field_names':
                            continue
                        
                        try:
                            # 测试值是否可以JSON序列化
                            json.dumps({key: value})
                            cleaned_metadata[key] = value
                        except (TypeError, OverflowError):
                            # 如果不能序列化，转换为字符串
                            cleaned_metadata[key] = str(value)
                    
                    # 添加额外的元素信息到元数据
                    cleaned_metadata['element_type'] = elem.__class__.__name__  # 元素类型名
                    cleaned_metadata['id'] = str(getattr(elem, 'id', None))  # 元素ID
                    cleaned_metadata['category'] = str(getattr(elem, 'category', None))  # 元素类别
                    
                    # 构建文本块数据结构
                    text_blocks.append({
                        "text": str(elem),  # 元素的文本内容
                        "page": page_number,  # 页码
                        "metadata": cleaned_metadata  # 清理后的元数据
                    })
            
            # 设置总页数（取页码集合中的最大值）
            self.total_pages = max(pages) if pages else 0
            # 保存当前页面映射，供后续使用
            self.current_page_map = text_blocks
            # 将所有文本块的内容用换行符连接返回
            return "\n".join(block["text"] for block in text_blocks)
            
        except Exception as e:
            # 记录错误日志并重新抛出异常
            logger.error(f"Unstructured error: {str(e)}")
            raise
    
    def _load_with_pdfplumber(self, file_path: str) -> str:
        """
        使用pdfplumber库加载PDF文档。
        适合需要处理表格或需要文本位置信息的场景。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"pdfplumber error: {str(e)}")
            raise
    
    def save_document(self, filename: str, chunks: list, metadata: dict, loading_method: str, strategy: str = None, chunking_strategy: str = None, output_format: str = "text") -> str:
        """
        保存处理后的文档数据。

        参数:
            filename (str): 原PDF文件名
            chunks (list): 文档分块列表
            metadata (dict): 文档元数据
            loading_method (str): 使用的加载方法
            strategy (str, optional): 使用的加载策略
            chunking_strategy (str, optional): 使用的分块策略
            output_format (str, optional): 输出格式

        返回:
            str: 保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            base_name = filename.replace('.pdf', '').split('_')[0]
            
            # Adjust the document name to include strategy and format
            format_suffix = f"_{output_format}" if output_format != "text" else ""
            if loading_method == "unstructured" and strategy:
                doc_name = f"{base_name}_{loading_method}_{strategy}_{chunking_strategy}{format_suffix}_{timestamp}"
            else:
                doc_name = f"{base_name}_{loading_method}{format_suffix}_{timestamp}"
            
            # 构建文档数据结构，确保所有值都是可序列化的
            document_data = {
                "filename": str(filename),
                "total_chunks": int(len(chunks)),
                "total_pages": int(metadata.get("total_pages", 1)),
                "loading_method": str(loading_method),
                "loading_strategy": str(strategy) if loading_method == "unstructured" and strategy else None,
                "chunking_strategy": str(chunking_strategy) if loading_method == "unstructured" and chunking_strategy else None,
                "output_format": str(output_format),
                "chunking_method": "loaded",
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks
            }
            
            # 保存到文件
            filepath = os.path.join("01-loaded-docs", f"{doc_name}.json")
            os.makedirs("01-loaded-docs", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise
    
    def _convert_to_markdown(self) -> str:
        """
        将当前页面映射转换为Markdown格式。
        
        使用PyMuPDF提取的结构化信息来生成更准确的Markdown格式。
        
        返回:
            str: Markdown格式的文档内容
        """
        if not self.current_page_map:
            return ""
        
        markdown_content = []
        current_page = None
        
        for block_data in self.current_page_map:
            block_text = block_data["text"]
            page_num = block_data["page"]
            structure_info = block_data.get("structure_info", {})
            
            # 添加页面分隔符（当页面变化时）
            if current_page is not None and current_page != page_num:
                markdown_content.append(f"\n---\n*Page {page_num}*\n")
            current_page = page_num
            
            # 根据结构信息处理内容
            processed_content = self._process_structured_block_to_markdown(block_text, structure_info)
            if processed_content:
                markdown_content.append(processed_content)
        
        # 清理和优化最终输出
        return self._clean_and_optimize_markdown("\n\n".join(markdown_content))
    
    def _process_structured_block_to_markdown(self, text: str, structure_info: Dict) -> str:
        """
        根据结构化信息将文本块转换为Markdown格式。
        
        参数:
            text (str): 文本内容
            structure_info (Dict): 结构化信息
            
        返回:
            str: Markdown格式的文本
        """
        if not text.strip():
            return ""
        
        # 如果是简单提取（降级模式），使用原有方法
        if structure_info.get("is_simple", False):
            return self._process_text_to_markdown(text)
        
        # 处理标题
        if structure_info.get("is_title", False):
            title_level = structure_info.get("title_level", 1)
            title_level = min(max(title_level, 1), 6)  # 限制在1-6级
            return f"{'#' * title_level} {text.strip()}"
        
        # 处理列表
        if structure_info.get("is_list", False):
            return self._convert_to_markdown_list(text)
        
        # 处理表格
        if structure_info.get("is_table", False):
            return self._convert_to_markdown_table(text)
        
        # 处理粗体文本
        if structure_info.get("has_bold", False) and len(text) < 200:
            # 短文本且为粗体，可能是重要信息
            return f"**{text.strip()}**"
        
        # 处理斜体文本
        if structure_info.get("has_italic", False) and len(text) < 200:
            # 短文本且为斜体，可能是注释或引用
            return f"*{text.strip()}*"
        
        # 默认作为段落处理
        return text.strip()
    
    def _convert_to_markdown_list(self, text: str) -> str:
        """
        将文本转换为Markdown列表格式。
        
        参数:
            text (str): 文本内容
            
        返回:
            str: Markdown列表格式
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否已经是markdown格式
            if re.match(r'^\s*[-*+]\s+', line):
                markdown_lines.append(line)
            # 转换有序列表
            elif re.match(r'^\s*\d+[\.\)]\s+', line):
                content = re.sub(r'^\s*\d+[\.\)]\s+', '', line)
                markdown_lines.append(f"- {content}")
            # 转换字母列表
            elif re.match(r'^\s*[a-zA-Z][\.\)]\s+', line):
                content = re.sub(r'^\s*[a-zA-Z][\.\)]\s+', '', line)
                markdown_lines.append(f"- {content}")
            # 转换其他列表标记
            elif re.match(r'^\s*[•·]\s+', line):
                content = re.sub(r'^\s*[•·]\s+', '', line)
                markdown_lines.append(f"- {content}")
            else:
                # 如果不是明显的列表格式，但在列表块中，添加列表标记
                markdown_lines.append(f"- {line}")
        
        return '\n'.join(markdown_lines)
    
    def _convert_to_markdown_table(self, text: str) -> str:
        """
        将文本转换为Markdown表格格式。
        
        参数:
            text (str): 文本内容
            
        返回:
            str: Markdown表格格式
        """
        lines = text.split('\n')
        table_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 如果已经是markdown表格格式
            if '|' in line and line.startswith('|') and line.endswith('|'):
                table_lines.append(line)
            # 制表符分隔
            elif '\t' in line:
                cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                if cells:
                    table_line = '| ' + ' | '.join(cells) + ' |'
                    table_lines.append(table_line)
            # 多个空格分隔（可能是对齐的列）
            elif re.search(r'\s{3,}', line):
                cells = [cell.strip() for cell in re.split(r'\s{3,}', line) if cell.strip()]
                if cells and len(cells) > 1:
                    table_line = '| ' + ' | '.join(cells) + ' |'
                    table_lines.append(table_line)
            else:
                # 尝试按空格分割（如果有多个单词）
                words = line.split()
                if len(words) > 2:  # 至少3个单词才考虑作为表格行
                    table_line = '| ' + ' | '.join(words) + ' |'
                    table_lines.append(table_line)
                else:
                    table_lines.append(line)  # 保持原样
        
        # 如果识别出表格行，添加表头分隔符
        if table_lines and any('|' in line for line in table_lines):
            # 找到第一个表格行，添加分隔符
            for i, line in enumerate(table_lines):
                if '|' in line:
                    # 计算列数
                    col_count = line.count('|') - 1
                    if col_count > 0:
                        separator = '|' + ' --- |' * col_count
                        table_lines.insert(i + 1, separator)
                    break
        
        return '\n'.join(table_lines)
    
    def _clean_and_optimize_markdown(self, markdown_text: str) -> str:
        """
        清理和优化Markdown文本。
        
        参数:
            markdown_text (str): 原始Markdown文本
            
        返回:
            str: 优化后的Markdown文本
        """
        if not markdown_text:
            return ""
        
        lines = markdown_text.split('\n')
        cleaned_lines = []
        prev_line_type = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # 确定行类型
            current_line_type = self._get_line_type(line_stripped)
            
            # 添加适当的空行
            if (prev_line_type and current_line_type and 
                prev_line_type != current_line_type and 
                prev_line_type != 'empty' and current_line_type != 'empty'):
                # 在不同类型的内容之间添加空行
                if cleaned_lines and cleaned_lines[-1].strip():
                    cleaned_lines.append("")
            
            # 避免连续的空行
            if not (line_stripped == "" and 
                   cleaned_lines and cleaned_lines[-1].strip() == ""):
                cleaned_lines.append(line)
            
            prev_line_type = current_line_type
        
        # 移除开头和结尾的空行
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _get_line_type(self, line: str) -> str:
        """
        获取行的类型。
        
        参数:
            line (str): 文本行
            
        返回:
            str: 行类型
        """
        if not line:
            return 'empty'
        elif line.startswith('#'):
            return 'heading'
        elif line.startswith('-') or line.startswith('*') or line.startswith('+'):
            return 'list'
        elif line.startswith('|') and line.endswith('|'):
            return 'table'
        elif line.startswith('>'):
            return 'quote'
        elif line.startswith('---'):
            return 'separator'
        else:
            return 'paragraph'
    
    def _process_text_to_markdown(self, text: str) -> str:
        """
        将单页文本内容处理为Markdown格式。
        
        参数:
            text (str): 原始文本内容
            
        返回:
            str: 处理后的Markdown文本
        """
        lines = text.split('\n')
        markdown_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue
            
            # 识别和转换不同类型的内容
            processed_line = self._identify_and_convert_line(line, i, lines)
            markdown_lines.append(processed_line)
        
        # 后处理：清理多余的空行
        return self._clean_markdown_output(markdown_lines)
    
    def _identify_and_convert_line(self, line: str, line_index: int, all_lines: List[str]) -> str:
        """
        识别并转换单行文本为相应的Markdown格式。
        
        参数:
            line (str): 当前行文本
            line_index (int): 行索引
            all_lines (List[str]): 所有行的列表（用于上下文分析）
            
        返回:
            str: 转换后的Markdown行
        """
        # 1. 检查是否为标题
        title_level = self._detect_title_level(line, line_index, all_lines)
        if title_level > 0:
            return f"{'#' * title_level} {line}"
        
        # 2. 检查是否为列表项
        if self._is_list_item(line):
            return self._convert_to_list_item(line)
        
        # 3. 检查是否为表格行
        if self._is_table_row(line):
            return self._convert_to_table_row(line)
        
        # 4. 检查是否为引用或特殊格式
        if self._is_quote_or_special(line):
            return self._convert_quote_or_special(line)
        
        # 5. 默认作为普通段落处理
        return line
    
    def _detect_title_level(self, line: str, line_index: int, all_lines: List[str]) -> int:
        """
        检测文本行是否为标题，并返回标题级别。
        
        使用多种启发式方法：
        1. 长度较短（通常少于80字符）
        2. 全大写或首字母大写
        3. 可能包含数字编号
        4. 后面跟着空行或内容
        5. 字体大小信息（如果可用）
        
        参数:
            line (str): 当前行文本
            line_index (int): 行索引
            all_lines (List[str]): 所有行的列表
            
        返回:
            int: 标题级别（1-6），0表示不是标题
        """
        # 基本过滤：空行或过长的行不太可能是标题
        if not line or len(line) > 100:
            return 0
        
        # 检查是否已经是markdown标题格式
        if line.startswith('#'):
            return 0  # 已经是markdown格式，不需要转换
        
        title_score = 0
        
        # 1. 长度评分（较短的行更可能是标题）
        if len(line) < 50:
            title_score += 2
        elif len(line) < 80:
            title_score += 1
        
        # 2. 大写字母评分
        if line.isupper():
            title_score += 3  # 全大写很可能是标题
        elif line.istitle() or line[0].isupper():
            title_score += 2  # 首字母大写
        
        # 3. 数字编号模式
        if re.match(r'^\d+\.?\s+', line) or re.match(r'^[IVX]+\.?\s+', line):
            title_score += 2  # 数字或罗马数字编号
        
        # 4. 章节关键词
        chapter_keywords = ['chapter', 'section', '章', '节', '部分', '第.*章', '第.*节']
        if any(re.search(keyword, line.lower()) for keyword in chapter_keywords):
            title_score += 2
        
        # 5. 上下文分析
        # 检查下一行是否为空行或内容行
        if line_index + 1 < len(all_lines):
            next_line = all_lines[line_index + 1].strip()
            if not next_line:  # 下一行为空
                title_score += 1
            elif len(next_line) > len(line):  # 下一行比当前行长（可能是内容）
                title_score += 1
        
        # 6. 特殊字符检查（标题通常不包含过多标点）
        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', line))
        if special_chars > len(line) * 0.3:  # 特殊字符超过30%
            title_score -= 2
        
        # 根据评分确定标题级别
        if title_score >= 6:
            return 1  # 一级标题
        elif title_score >= 4:
            return 2  # 二级标题
        elif title_score >= 3:
            return 3  # 三级标题
        else:
            return 0  # 不是标题
    
    def _is_list_item(self, line: str) -> bool:
        """检查是否为列表项。"""
        # 检查各种列表格式
        list_patterns = [
            r'^\s*[-*+]\s+',  # 无序列表
            r'^\s*\d+\.\s+',  # 有序列表
            r'^\s*[a-zA-Z]\.\s+',  # 字母列表
            r'^\s*[IVX]+\.\s+',  # 罗马数字列表
        ]
        return any(re.match(pattern, line) for pattern in list_patterns)
    
    def _convert_to_list_item(self, line: str) -> str:
        """转换为Markdown列表格式。"""
        # 如果已经是markdown格式，直接返回
        if re.match(r'^\s*[-*]\s+', line):
            return line
        
        # 转换有序列表为无序列表（可根据需要调整）
        if re.match(r'^\s*\d+\.\s+', line):
            content = re.sub(r'^\s*\d+\.\s+', '', line)
            return f"- {content}"
        
        # 转换其他格式
        if re.match(r'^\s*[a-zA-Z]\.\s+', line):
            content = re.sub(r'^\s*[a-zA-Z]\.\s+', '', line)
            return f"- {content}"
        
        return line
    
    def _is_table_row(self, line: str) -> bool:
        """检查是否为表格行。"""
        # 检查是否包含表格分隔符
        return '|' in line or '\t' in line or re.search(r'\s{3,}', line)
    
    def _convert_to_table_row(self, line: str) -> str:
        """转换为Markdown表格格式。"""
        if '|' in line:
            # 已经是markdown表格格式
            return line
        elif '\t' in line:
            # 制表符分隔，转换为markdown表格
            cells = line.split('\t')
            return '| ' + ' | '.join(cell.strip() for cell in cells) + ' |'
        else:
            # 空格分隔，尝试识别列
            # 这里使用简单的空格分割，实际应用中可能需要更复杂的逻辑
            cells = re.split(r'\s{3,}', line)
            if len(cells) > 1:
                return '| ' + ' | '.join(cell.strip() for cell in cells) + ' |'
        
        return line
    
    def _is_quote_or_special(self, line: str) -> bool:
        """检查是否为引用或特殊格式。"""
        # 检查引用标记
        quote_patterns = [
            r'^\s*["""'']\s*',  # 引号开始
            r'^\s*>\s+',  # 引用符号
            r'^\s*注[：:]\s*',  # 注释
            r'^\s*备注[：:]\s*',  # 备注
        ]
        return any(re.match(pattern, line) for pattern in quote_patterns)
    
    def _convert_quote_or_special(self, line: str) -> str:
        """转换引用或特殊格式。"""
        if re.match(r'^\s*>\s+', line):
            return line  # 已经是markdown引用格式
        
        if re.match(r'^\s*["""'']\s*', line):
            content = re.sub(r'^\s*["""'']\s*', '', line)
            return f"> {content}"
        
        if re.match(r'^\s*注[：:]\s*', line):
            content = re.sub(r'^\s*注[：:]\s*', '', line)
            return f"> **注：** {content}"
        
        if re.match(r'^\s*备注[：:]\s*', line):
            content = re.sub(r'^\s*备注[：:]\s*', '', line)
            return f"> **备注：** {content}"
        
        return line
    
    def _clean_markdown_output(self, lines: List[str]) -> str:
        """
        清理Markdown输出，移除多余的空行并优化格式。
        
        参数:
            lines (List[str]): Markdown行列表
            
        返回:
            str: 清理后的Markdown文本
        """
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = not line.strip()
            
            # 避免连续的空行
            if is_empty and prev_empty:
                continue
            
            cleaned_lines.append(line)
            prev_empty = is_empty
        
        # 移除开头和结尾的空行
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
