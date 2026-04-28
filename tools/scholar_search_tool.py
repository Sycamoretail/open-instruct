import json
import requests
import time
import dateutil
import datetime
import xml.etree.ElementTree as ET
import re
# from byted_doubaoagent_adapter import SeedApplicationAiSearchPromptAdapter

class ScholarSearchTool:
    name = "ScholarSearch"
    description = ""
    def __init__(self, return_prompt_type="text", adapter=None):
        super().__init__()
        self.return_prompt_type = return_prompt_type
        self.adapter = adapter
        self.ignore_image = True
        self.api_key = "d294932a-248c-44d4-93b6-13b65e00f3e4"
        #self.url = "https://gpt.bytedance.net/gpt/tool_hub/online/apihub/function_call_proxy"
        self.url = "https://gpt.bytedance.net/gpt/tool_hub/online/mcp_server/proxy/global_search_v2/mcp"
        self.headers = { "api-key": self.api_key, "Content-Type": "application/json" }
        self.is_card = lambda doc: doc.get("ruyi_info", {}).get("ruyi_type", "") in [
            "moji_weather",
            "future_weather",
            "exchange_rate",
            "tt_stock",
            "hanzi",
            "aft_hanzi_detail",
            "calendar_new",
            "dict",
            "tt_fund",
            "futures_pic",
            "forex_trend",
            "gold_trend",
        ]
    
    def get_response(self, search_request_list: list, last_time: str=None):
        for req in search_request_list:
            req['pagination'] = {"limit":8}    # 一个query8个doc
            if last_time is not None:
                req['publish_end_date'] = last_time
        #input_params = json.dumps({"search_request_list":search_request_list})
        #data = {"name": "ScholarSearch", "input_params": input_params, "api_id": "6313"}
        data = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "ScholarSearch",
                "arguments": {
                    "search_request_list": search_request_list
                }
            }
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code != 200:
            return None, {"status_message": f"Failed to call API Hub: {response.status_code} {response.text} {response.headers}"}
        log_id, result = response.headers.get("X-Tt-Logid"), response.json()
        #search_result = json.loads(result.get("data", {}).get("result", {}))
        search_result = json.loads(result['result']['content'][0]['text'])['result']
        return search_result, {"status_message": "success", "status_code": 0, "log_id": log_id}
    
    def parse_publish_time(self, publish_time_str):
        publish_time = ""
        try:
            if len(publish_time_str) > 0:
                publish_time = dateutil.parser.parse(publish_time_str) 
                time_var = {
                    "year": publish_time.year,
                    "month": publish_time.month,
                    "day": publish_time.day,
                    "hour": publish_time.hour,
                    "minute": publish_time.minute
                }
                publish_time = "{year}年{month}月{day}日{hour}时{minute}分".format(**time_var)
        except (ValueError, TypeError):
            publish_time = ""
        return publish_time
    
    def adapt_to_text(self, search_result: dict, page_idx: int, url_tosurl_map: dict, tosurl_url_map: dict, no_display_url_map: dict, url_no_display_map: dict):
        rsp_str = ''
        for search_response in search_result['search_response_list']:
            if search_response.get('status_code', -1) != 0:  # 没有字段也是失败
                continue
            for doc in search_response['documents']:  # 工具内部已经有截断(每个query返回最大10个)，这里和线上保持一致，不做额外截断
                doc_info = doc["doc_info"]
                if len(doc_info.get("tos_url", "")) > 0:  # 保存url -> tos_url映射
                    url_tosurl_map[doc_info.get("url", "")] = doc_info.get("tos_url", "")
                    tosurl_url_map[doc_info.get("tos_url", "")] = doc_info.get("url", "")
                if doc["display_info"]["no_display"]: # 不可见url处理 -> 真实url
                    doc_id = doc.get("doc_id", "")
                    url = "http://www.shurl.cc/" + doc_id.replace('-', '')
                    no_display_url_map[url] = doc_info.get("url", "")
                    url_no_display_map[doc_info.get("url", "")] = url
                else:
                    url = doc_info.get("url", "")
                
                rsp_str += "<|superscript|>:{}：\n标题：{}\n作者：{}\n摘要：{}\n被引用量：{}\n发表源：{}\n数据来源：{}\n链接：{}\n发表日期：{}\n".format(
                    page_idx+1, 
                    doc_info.get("title", ""), 
                    ";".join([item.get("name", "") for item in doc.get("author_info", [])]), 
                    doc_info.get("overview", "").strip(),  # 摘要
                    doc.get("statistic_info", {}).get("cite_count", ""), 
                    doc.get("host_info", {}).get("hostname", ""), 
                    doc_info.get("publisher", ""),
                    url,
                    self.parse_publish_time(doc_info.get("publish_time", ""))
                )  
                page_idx += 1
        return rsp_str, page_idx 

    def _protect_all_text_data(self, element, temp_replacement, placeholder_prefix="__PLACEHOLDER_"):
        """
        递归地保护XML元素中的所有文本数据，包括元素文本内容和属性值
        跳过已经被原有保护机制处理的内容（以||开头和结尾的占位符）
        """
        placeholder_counter = len(temp_replacement)
        
        # 保护元素的文本内容
        if element.text and element.text.strip():
            # 跳过已经被原有机制保护的内容（||text_content||N, ||image_uri||xxx, 等）
            if not (element.text.startswith("||") and element.text.count("||") >= 2):
                placeholder = f"{placeholder_prefix}{placeholder_counter}_TEXT__"
                temp_replacement[placeholder] = element.text
                element.text = placeholder
                placeholder_counter += 1
        
        # 保护元素的尾部文本（在XML中通常不需要，但为了完整性）
        if element.tail and element.tail.strip():
            if not (element.tail.startswith("||") and element.tail.count("||") >= 2):
                placeholder = f"{placeholder_prefix}{placeholder_counter}_TAIL__"
                temp_replacement[placeholder] = element.tail
                element.tail = placeholder
                placeholder_counter += 1
        
        # 保护所有属性值
        for attr_name, attr_value in list(element.attrib.items()):
            if attr_value.strip():  # 只保护非空属性值
                # 跳过已经被原有机制保护的reference_id等
                if not (attr_value.startswith("<|") and attr_value.endswith("|>") and ":" in attr_value):
                    placeholder = f"{placeholder_prefix}{placeholder_counter}_ATTR_{attr_name.upper()}__"
                    temp_replacement[placeholder] = attr_value
                    element.set(attr_name, placeholder)
                    placeholder_counter += 1
        
        # 递归处理子元素
        for child in element:
            self._protect_all_text_data(child, temp_replacement, placeholder_prefix) 
    
    def _tranform_search_result_doc(self, function_name, search_document, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map):
        ref_type = "<|superscript|>:"
        if function_name == "GeneralSearch":
            if self.is_card(search_document):
                ref_type = "<|card|>:"
        result = [("ref_type", ref_type)]

        # doc overview information: title/summary
        title = search_document.get("doc_info", {}).get("title", "")
        if len(title) > 0:
            result.append(("title", title))
    
        summary = search_document.get("doc_info", {}).get("overview", "")
        if ref_type == "<|card|>:":
            summary = search_document.get("ruyi_info", {}).get("description") 

        if len(summary) > 0:
            result.append(("summary", summary))

        # doc meta information: host/publisher/author
        host = search_document.get("host_info", {}).get("hostname", "")
        if len(host) > 0:
            result.append(("hostname", host))

        publisher = search_document.get("doc_info", {}).get("publisher", "")
        if len(publisher) > 0:
            result.append(("publisher", publisher))

        cite_count = search_document.get("statistic_info", {}).get("cite_count", -1)
        if cite_count >= 0:
            result.append(("cite_count", f"{cite_count}"))

        author = search_document.get("author_info", [])
        if len(author) > 0:
            result.append(("author", author))

        # doc content information: 
        if function_name != "ScholarSearch":
            snippet = search_document.get("doc_info", {}).get("snippet", [])
            if len(snippet) > 0:
                result.append(("snippet", snippet))

        url = search_document.get("doc_info", {}).get("url", "")

        # ruyi的落地页，fetch不可读，先不拼接
        if len(url) > 0 and search_document.get("doc_info", {}).get("filetype", "") != "ruyi":
            no_display = search_document.get("display_info", {}).get("no_display", False)
            if no_display:      
                doc_id = search_document.get("doc_id", "")
                if len(doc_id) > 0:
                    doc_id = doc_id.replace("-", "")
                    short_url = f"http://www.shurl.cc/{doc_id}"
                    result.append(("url", short_url)) 
                    no_display_url_map[short_url] = url
                    url_no_display_map[url] = short_url
            else:
                result.append(("url", url))
            tos_url = search_document.get("doc_info", {}).get("tos_url", "")
            if len(tos_url) > 0:
                url_tosurl_map[url] = tos_url
                tosurl_url_map[tos_url] = url

        publish_time = ""
        publish_time_str = search_document.get("doc_info", {}).get("publish_time", "")
        try:
            if len(publish_time_str) > 0:
                publish_time = dateutil.parser.parse(publish_time_str) 
                time_var = {
                    "year": publish_time.year,
                    "month": publish_time.month,
                    "day": publish_time.day,
                    "hour": publish_time.hour,
                    "minute": publish_time.minute
                }
                publish_time = "{year}年{month}月{day}日{hour}时{minute}分".format(**time_var)
        except (ValueError, TypeError):
            publish_time = ""
        if len(publish_time) > 0:
            result.append(("publish_time", publish_time))
            
        if host == "arXiv":
            result.append(("arxiv_url", url))
            # 提取arXiv ID
            # 支持：
            # - https://arxiv.org/pdf/2305.12345.pdf
            # - https://arxiv.org/html/2305.12345
            # - https://arxiv.org/abs/2305.12345
            # - https://arxiv.org/abs/cs/9901001
            arxiv_match = re.search(r'/(?:pdf|html|abs)/([^?#]+)', url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                # 移除可能的查询参数/fragment
                arxiv_id = arxiv_id.split('?', 1)[0].split('#', 1)[0]
                # 去掉常见的 .pdf 后缀
                if arxiv_id.endswith('.pdf'):
                    arxiv_id = arxiv_id[:-4]
                # 兼容旧逻辑可能多抓一个末尾点（例如 2305.12345.）
                arxiv_id = arxiv_id.rstrip('.')
                result.append(("arxiv_id", arxiv_id))
            
        return result
    
    def _format_search_time(self, search_time):
        if not search_time:
            search_time = datetime.datetime.now()
        
        time_var = {
            "year": search_time.year,
            "month": search_time.month,
            "day": search_time.day,
            "hour": search_time.hour,
            "minute": search_time.minute
        }
        search_time_str = "{year}年{month}月{day}日{hour}时{minute}分".format(**time_var)
        search_time_str += {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日",
        }.get(search_time.weekday(), "")
        return search_time_str
    
    def _transform_to_output(self, root, temp_replacement=None):
        # 创建root的深拷贝，避免修改原始XML树
        import copy
        protected_root = copy.deepcopy(root)
        
        # 如果没有传入temp_replacement，创建一个空的字典
        if temp_replacement is None:
            temp_replacement = {}
        
        # 合并传入的temp_replacement和新创建的保护字典
        all_replacements = dict(temp_replacement)  # 复制原有的替换字典
        
        # 保护XML树中的所有文本数据
        self._protect_all_text_data(protected_root, all_replacements)
        
        # 将保护后的XML转换为字符串并美化格式
        rough_content = ET.tostring(protected_root, 'utf-8', xml_declaration=True)
        lines = rough_content.decode('utf-8').split("\n")
        
        results = list()
        for line in lines:
            if line.strip() and not line.strip().startswith('<?xml'):
                # 把转译后的reference_id（保留原有逻辑，防止某些特殊情况）
                line = line.replace("&lt;|superscript|&gt;:", "<|superscript|>:")
                line = line.replace("&lt;|card|&gt;:", "<|card|>:")
                line = line.replace("&lt;|image|&gt;:", "<|image|>:")
                line = line.replace("&lt;|link|&gt;:", "<|link|>:")
                
                # 把所有占位符替换回原始内容（按长度降序排序，避免短占位符被误替换）
                for placeholder, original_content in sorted(all_replacements.items(), key=lambda x: len(x[0]), reverse=True):
                    if placeholder in line:
                        line = line.replace(placeholder, original_content)
                results.append(line)
        return "\n".join(results)
    
    def _filter_content_part(self, content):
        if not self.ignore_image:
            return content
        merged_text = ""
        result = []
        for item in content:
            if item.get("type", "") == "text":
                merged_text += item.get("text", "")
            elif item.get("type", "") == "image":
                continue
            else:
                result.append({"type": "text", "text": merged_text})
                merged_text = ""
                result.append(item)
        if len(merged_text) > 0:
            result.append({"type": "text", "text": merged_text})
        return result

    def _process_mixed_content(self, parent_element, content_list, base_reference_id, content_source_type):
        """
        Process mixed content (text, image, link) and add them to parent element with proper text/tail handling
        
        This function implements a key optimization to avoid generating unnecessary <text></text> tags:
        
        - For single text content: Sets parent_element.text directly
          ✅ Generates: <snippet>Simple text</snippet>
          ❌ Avoids:   <snippet><text>Simple text</text></snippet>
        
        - For mixed content: Uses XML text/tail attributes for proper text positioning
          ✅ Generates: <snippet>Start <image>...</image> between <link>...</link> end</snippet>
        
        Args:
            parent_element: XML element to add content to
            content_list: List of content items (text, image, link)
            base_reference_id: Base reference ID for generating child reference IDs
            content_source_type: Type of content source ('search' or 'fetch')
        
        Returns:
            None (modifies parent_element in place)
        """
        filtered_content = self._filter_content_part(content_list)
        
        # Key optimization: Check if content contains only text
        text_only = all(item.get("type", "") == "text" for item in filtered_content)
        
        if text_only and len(filtered_content) == 1:
            # Single text optimization: Avoid creating unnecessary <text> tags
            # Direct assignment to parent_element.text for cleaner XML structure
            parent_element.text = filtered_content[0].get("text", "")
        else:
            # Mixed content handling: Use XML text/tail attributes for proper positioning
            current_text = ""
            last_element = None
            image_idx = 1
            link_idx = 1
            
            for item in filtered_content:
                if item.get("type", "") == "text":
                    # Accumulate text content for proper positioning
                    current_text += item.get("text", "")
                elif item.get("type", "") == "link":
                    # Handle text positioning: text before this link element
                    if current_text:
                        if last_element is None:
                            # First element: text goes to parent.text
                            parent_element.text = current_text
                        else:
                            # After previous element: text goes to last_element.tail
                            last_element.tail = current_text
                        current_text = ""
                    
                    link_ref_id = f"{base_reference_id}-<|link|>:{link_idx}"
                    # Auto-append element type to content source type
                    link_source_type = f"{content_source_type}_link"
                    link_idx += 1
                    
                    link_et = ET.SubElement(parent_element, "link")
                    link_et.set("reference_id", link_ref_id)
                    link_et.set("text", item.get("text", ""))
                    last_element = link_et
            
            # Handle any remaining text content at the end
            if current_text:
                if last_element is None:
                    # No elements were created: all text goes to parent.text
                    parent_element.text = current_text
                else:
                    # After last element: text goes to last_element.tail
                    last_element.tail = current_text
    
    def _single_search_result_prompt(self, function_name, search_time, search_request, search_response, page_idx, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title):
        """
        将搜索结果转换为指定格式的XML字符串
        
        参数:
            query: 搜索关键词
            search_result: 包含搜索结果的字典
            
        返回:
            符合指定格式的XML字符串
        """
        # 创建根元素
        root = ET.Element("search")
        
        # TODO: VisualSearch的query是什么？
        root.set("query", search_request.get("query", ""))
        
        root.set("search_time", self._format_search_time(search_time))

        # Check for search-level error first
        status_code = search_response.get("status_code", 0)
        status_message = search_response.get("status_message", "success")
        
        if status_code != 0 and len(status_message) > 0 and status_message != "success":
            err_msg_et = ET.SubElement(root, "err_msg")
            err_msg_et.text = status_message
        
        # 遍历所有文档
        for i, doc in enumerate(search_response.get("documents", [])):
            if curr_title and curr_title[:15].lower().replace("’", "'") in doc.get("doc_info", {}).get("title", "").lower().replace("’", "'"):
                continue
            # 创建document元素
            document = ET.SubElement(root, "document") 

            transformed_doc = self._tranform_search_result_doc(function_name, doc, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map)
            for key, value in transformed_doc:
                if len(value) == 0:
                    continue

                if key == "ref_type":
                    reference_type = value
                    current_id = page_idx+1
                    page_idx += 1
                    
                    reference_id = f"{reference_type}{current_id}"

                    document.set("reference_id", reference_id)
                elif key == "snippet":  
                    # 提取snippet文本内容
                    snippet_et = ET.SubElement(document, "snippet")
                    self._process_mixed_content(snippet_et, value, reference_id, "search")

                elif key == "author":
                    author_list_et = ET.SubElement(document, "author_list")
                    for author in value:
                        name, affiliation = author.get("name", ""), author.get("affiliation", "")
                        if len(name) > 0:
                            author_et = ET.SubElement(author_list_et, "author")
                            author_et.set("name", author.get("name"))
                            if len(affiliation) > 0:
                                author_et.set("affiliation", affiliation)
                else:
                    ET.SubElement(document, key).text = value
        
        # 将XML转换为字符串并美化格式
        return self._transform_to_output(root), page_idx
    
    def adapt_to_xml(self, search_request_list: list, search_result: dict, page_idx: int, url_tosurl_map: dict, tosurl_url_map: dict, no_display_url_map: dict, url_no_display_map: dict, curr_title: str):
        results = list()
        search_time = datetime.datetime.now()
        
        # 检查search_result是否包含必要的键
        if 'search_response_list' not in search_result:
            return "搜索结果格式错误", page_idx
        
        # 确保search_response_list和search_request_list长度匹配
        search_response_list = search_result['search_response_list']
        if len(search_response_list) != len(search_request_list):
            # 使用zip_shortest确保即使长度不匹配也能处理所有请求
            from itertools import zip_longest
            items = zip_longest(search_request_list, search_response_list, fillvalue={})
        else:
            items = zip(search_request_list, search_response_list)
            
        for search_request, search_response in items:
            content, page_idx = self._single_search_result_prompt("ScholarSearch", search_time, search_request, search_response, page_idx, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title)
            results.append(content)
        return "\n".join(results), page_idx

    def __call__(self, search_request_list, page_idx=None, last_time=None, curr_title=None, refid_url_map=None, no_display_url_map=None, url_tosurl_map=None, tosurl_url_map=None, url_no_display_map=None):
        # 初始化可选参数
        page_idx = page_idx or 0
        refid_url_map = refid_url_map or {}
        no_display_url_map = no_display_url_map or {}
        url_tosurl_map = url_tosurl_map or {}
        tosurl_url_map = tosurl_url_map or {}
        url_no_display_map = url_no_display_map or {}
        
        search_result, metrics = self.get_response(search_request_list, last_time)

        # 检查API调用是否成功
        if search_result is None:
            print('---------------------------')
            rsp_str = "API调用失败，请稍后重试"
            return {
                "content": rsp_str,
                "metric": metrics,
                "page_idx": page_idx
            }

        if self.return_prompt_type == 'text':
            rsp_str, page_idx = self.adapt_to_text(search_result, page_idx, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map)
        else:
            rsp_str, page_idx = self.adapt_to_xml(search_request_list, search_result, page_idx, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title)
            # search_time = datetime.datetime.now()
            # tool_prompt, _ = self.adapter.get_search_result_prompt_and_refids(
            #     "ScholarSearch", search_time, {"search_request_list": search_request_list}, search_result
            # )
            # rsp_str = ""
            # for each_tool_prompt in tool_prompt:
            #     if each_tool_prompt["type"] == "text":
            #         rsp_str += each_tool_prompt["text"]

        final_result = {
            "content": rsp_str,
            "metric": metrics,
            "page_idx": page_idx
        }
        return final_result


def uni_test():
    # adapter = SeedApplicationAiSearchPromptAdapter(ignore_image=True, biz_id="search_cot")
    scholar_search_tool = ScholarSearchTool(return_prompt_type="xml")
    search_request_list = [{"query":"烟草叶片瞬时转染 异源表达 启动子活性 可靠性 文献"}]
    page_idx = 0
    result = scholar_search_tool(search_request_list, page_idx, last_time="2026-03-31")
    print(result["content"])
    # print(result["url_to_papar_id"])
    # print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    uni_test()