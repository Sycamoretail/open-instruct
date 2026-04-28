import json
import os
import requests
import threading
import time
# from dr_tools.model import LLMModelClient
# from dr_tools.mcp_client_new import MCPClient
from transformers import AutoTokenizer
# import logid
import copy
import xml.etree.ElementTree as ET
# from byted_doubaoagent_adapter import SeedApplicationAiSearchPromptAdapter

# mcp_client = MCPClient("xxx","mcp_search_tool_config","NLP_LLM","rlhf_deep_research","ppe_browser_evaluate")

# class LinkReaderPluginTool:
#     name = "LinkReader"
#     description = """工具"精读"（LinkReader）介绍：
# - 功能说明：这是一个链接浏览工具，可以打开链接（链接可以是网页或pdf）并根据需求描述汇总页面上的所有相关信息。
# - 使用场景：当你需要进一步获取某个链接下有价值的信息时，可以调用本工具去精读该链接。建议的使用场景包括但不限于如下几种情况：1.任务中明确提供了网址需要精读获取信息，2.已经调用的工具（如搜索结果、之前精读网页返回的内容）返回了网址，且还需要从中进一步获取信息。请注意，尽量避免自己凭空构造链接并访问，这对于完成任务没有任何帮助。
# - 输入要求：你需要输入一个目标链接，以及一段需求描述。
# 1. 目标链接：要求来自已知内容（如工具返回、用户提供），如果不完整需要补全（以http开头）。
# 2. 需求描述：需要包括你想要从该链接中获取什么信息，需求描述应该清晰准确。请注意！本工具只会抽取已有的信息，不会进行额外的分析，因此需求应该贴合原文内容，否则什么也不会拿到！例如：对于一篇政策的链接，错误示范：需求描述中要求获取政策解读。正确示范：需求描述中要求获取政策中某部分的内容。
# - 调用格式示例：
# <action>
# {
#     "name": "LinkReader",
#     "arguments": {
#         "description": "需求描述",
#         "url": "目标链接"
#     },
# }
# </action>
# """
#     def __init__(self, min_summary_length=5000, link_mask_url=True, use_mcp=True, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         if model_type == "deepseek-r1":
#             self.model = LLMModelClient(model_type="deepseek-r1")
#         else:
#             self.model = LLMModelClient(model_type="doubao", model_psm=model_psm, model_cluster=model_cluster, temperature=0.01)
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode", temperature=0.01)  # 15b
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.sed.webgpt_100b_debug", model_cluster="default", temperature=0.01)  # 2b5
#         self.min_summary_length = min_summary_length
#         self.link_mask_url = link_mask_url
#         self.use_mcp = use_mcp

#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         # if os.path.exists(tokenizer_path):
#         #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         # else:
#         #     self.tokenizer = None

#     def summary_reference(self, content, description):
#         final_input = """你是一名信息收集专家，你的团队正在合作解决一个很难的问题，你的任务是从提供的 <长文本>（可能是文章或网页内容）中，尽可能准确地提取与 <需求描述> 相关的原文片段。提取结果会在后续阶段对解决问题提供帮助。
# 请严格遵循以下规则：
# 1. 尽可能准确地提取信息：一定不要一句话总结！另外如果文本中有多个相关片段，请全部提取，而不是仅选择一个。
# 2. 保持完整性：如果某个片段部分内容与需求相关，则保留整个片段，不要只截取部分句子。
# 3. 包含相关链接：部分文本有对应的链接（可能是完整链接，或是链接标记如"url1"），如果某个链接和需求可能相关，请提取该链接及其周围相关内容，确保信息完整。如果片段中包含了引用，请同时贴上相关引用的链接；
# 4. 保持原文语言：提取的内容必须与原文语言一致，不要进行翻译或改写。
# 5. 即使没有完全对应的内容，也尽力提取可能相关的信息，这对于解决问题很有帮助。
# 6. 直接输出答案，不要有多余内容。保持准确，信息量丰富和完整，结果在1000字左右即可。

# <长文本>
# {reference}
# </长文本>

# <需求描述>
# {task}
# </需求描述>

# 提取的原文片段如下：

# """
#         input_ids = self.tokenizer(content)["input_ids"]
#         max_chunk_size = 15000
#         overlap = 500
#         num_chunks = (len(input_ids) + max_chunk_size - 1) // max_chunk_size
#         num_chunks = min(num_chunks, 5)
#         summary_parts = []

#         for i in range(num_chunks):
#             start_idx = max(0, i * max_chunk_size - overlap)
#             end_idx = min((i + 1) * max_chunk_size + overlap, len(input_ids))
#             chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])
#             prompt = final_input.format(
#                 task=description, reference=chunk
#             )
#             messages = [{"role": "user", "content": prompt}]
#             response = self.model(messages)
#             response = response.split("</think>", 1)[-1].strip()
#             if len(response) > 0:
#                 summary_parts.append(response)
#         if len(summary_parts) == 1:
#             response = summary_parts[0]
#         else:
#             response = ""
#             for i, summary_part in enumerate(summary_parts):
#                 response += f"### 第{i+1}片段抽取结果：\n{summary_part}\n"
#         return response

#     def get_url_content_from_mcp(self, url):
#         arguments = {
#             "url": url,
#             "type": "全文",
#             "skip_cache": False,
#             "from_mcp_call": True,
#             "is_offline": True,
#         }
#         arguments_str = json.dumps(arguments)
#         data = {
#             "name": "LinkReader",
#             "arguments": arguments_str,
#             "traffic_group": "NLP_LLM",
#             "traffic_id": "rlhf",
#         }
#         for _ in range(3):
#             try:
#                 resp = requests.post(
#                     "https://bytemcp.bytedance.net/plugin/search/tools/call",
#                     headers={
#                         "X-Tt-Env": "ppe_deep_research",
#                         "X-Use-Ppe": "1",
#                         "Content-Type": "application/json",
#                     },
#                     json = data
#                 )
#                 logid = resp.headers.get("x-tt-logid")
#                 resp.raise_for_status()
#                 result = resp.json()
#                 result = json.loads(result["result"])
#                 if "data" in result and "content" in result["data"]:
#                     return result["data"]["content"], logid
#                 elif "pdf_content" in result:
#                     return result["pdf_content"], logid
#                 else:
#                     return "", logid
#             except:
#                 time.sleep(1)
#         return "", ""


#     def get_url_content(self, url):
#         thought = {
#             "action": "browser_navigate",
#             "url": url,
#             "need_element_analyzer": False,
#         }

#         body = {
#             "PluginThoughtList": [
#                 {
#                     "PluginName": "",
#                     "Thought": json.dumps(thought, ensure_ascii=False).encode("utf8").decode(),
#                     "ToolName": "BrowserUseAgent",
#                     "PluginId": "100072706",
#                 }
#             ],
#             "BizId": "seed",
#             "TrafficGroup": "doubao",
#             "TrafficId": "deep_research",
#             "ak": "cZ33F9UtXDvv3qBFU1cusuRmhbz51xnZ",
#         }
#         headers = {"Content-Type": "application/json;charset=UTF-8", "x-tt-env": "ppe_20250311210028", "x-use-ppe": "1"}
#         full_text = ""
#         logid = None
#         url_dict = {}
#         for _ in range(3):
#             try:
#                 resp = requests.post("https://ah3yte2a.fn.bytedance.net/api/v1/observe?", json=body, headers=headers).json()
#                 logid = resp["data"]["plugin_request"]["Base"]["LogID"]
#                 obs_list = resp.get("data", {}).get("plugin_observation_list", [])
#                 observation = json.loads(obs_list[0]["Observation"])
#                 web_page = observation["web_page_info_list"][0]
#                 content_block_list = web_page["content_info"]["content_block_list"]
#                 if content_block_list is not None and len(content_block_list) > 0:
#                     full_text = ""
#                     for text_piece in content_block_list:
#                         if len(text_piece["link"]) == 0:
#                             full_text += text_piece["text"]
#                         elif len(text_piece["text"]) > 0:
#                             k = f"(url{len(url_dict)+1})"
#                             url_dict[k] = text_piece["link"]
#                             if self.link_mask_url:
#                                 full_text += "{}".format(text_piece["text"])
#                             else:
#                                 full_text += "{}{}".format(text_piece["text"], k)
#                 else:
#                     full_text = web_page["content_info"]["content"]
#                 break
#             except:
#                 full_text = ""
#                 url_dict = {}
#                 time.sleep(3)
#         return full_text, url_dict, logid

#     def __call__(self, description: str, url: str):
#         begin_time = time.time()
#         if self.use_mcp:
#             full_text, logid = self.get_url_content_from_mcp(url)
#             url_dict = {}
#         else:
#             full_text, url_dict, logid = self.get_url_content(url)

#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             try:
#                 input_ids = self.tokenizer(full_text)["input_ids"]
#                 metric["link_reader_raw_content_len"] = len(input_ids)
#                 if len(input_ids) < self.min_summary_length:
#                     content = f"""# Page url: {url}\n# Page Summary:\n{full_text}\n"""
#                     metric["link_reader_summary"] = False
#                 else:
#                     response = self.summary_reference(full_text, description)
#                     for k, v in url_dict.items():
#                         if k in response:
#                             response = response.replace(k, "({})".format(v))
#                     rsp_ids = self.tokenizer(response)["input_ids"]
#                     if len(rsp_ids) > 3000:
#                         response = self.tokenizer.decode(rsp_ids[:3000]) + "..."
#                     content = f"""# Page url: {url}\n# Page Summary:\n{response}\n"""
#                     metric["link_reader_summary"] = True
#                 metric["browser_navigate_status"] = "success"
#             except Exception:
#                 content = f"Failed to browse content in {url}."
#                 metric["browser_navigate_status"] = "failed"
#                 metric["fialed_stage"] = "summary"
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_summary_time"] = end_time - browser_navigate_end_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])
#         result = {
#             "content": content,
#             "metric": metric,
#         }
#         return result


# class LinkReaderPluginToolV4:
#     name = "LinkReader"
#     description = """工具"精读"（LinkReader）介绍：
# - 功能说明：这是一个链接浏览工具，可以打开链接（链接可以是网页或pdf）并根据需求描述汇总页面上的所有相关信息。
# - 使用场景：当你需要进一步获取某个链接下有价值的信息时，可以调用本工具去精读该链接。建议的使用场景包括但不限于如下几种情况：1.任务中明确提供了网址需要精读获取信息，2.已经调用的工具（如搜索结果、之前精读网页返回的内容）返回了网址，且还需要从中进一步获取信息。请注意，尽量避免自己凭空构造链接并访问，这对于完成任务没有任何帮助。
# - 输入要求：你需要输入希望精读的目标链接（可多个），以及一段需求描述。
# 1. 目标链接：要求来自已知内容（如工具返回、用户提供），如果不完整需要补全（以http开头），如果希望从多篇url中获取相同的需求信息，可以填入多个目标链接。
# 2. 需求描述：需要包括你想要从这些链接中获取什么信息，需求描述应该清晰准确。请注意！本工具只会抽取已有的信息，不会进行额外的分析，因此需求应该贴合原文内容，否则什么也不会拿到！例如：对于一篇政策的链接，错误示范：需求描述中要求获取政策解读。正确示范：需求描述中要求获取政策中某部分的内容。
# - 调用格式示例：
# <action>
# {
#     "name": "LinkReader",
#     "arguments": {
#         "description": "需求描述",
#         "url": ["目标链接1", "目前链接2"]
#     },
# }
# </action>
# """
#     def __init__(self, min_summary_length=5000, link_mask_url=True, use_mcp=True, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         if model_type == "deepseek-r1":
#             self.model = LLMModelClient(model_type="deepseek-r1")
#         else:
#             self.model = LLMModelClient(model_type="doubao", model_psm=model_psm, model_cluster=model_cluster, temperature=0.01)
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode", temperature=0.01)  # 15b
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.sed.webgpt_100b_debug", model_cluster="default", temperature=0.01)  # 2b5
#         self.min_summary_length = min_summary_length
#         self.link_mask_url = link_mask_url
#         self.use_mcp = use_mcp

#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         # if os.path.exists(tokenizer_path):
#         #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         # else:
#         #     self.tokenizer = None

#     def summary_reference(self, content, description):
#         final_input = """你是一名信息收集专家，你的团队正在合作解决一个很难的问题，你的任务是从提供的 <长文本>（可能是文章或网页内容）中，尽可能准确地提取与 <需求描述> 相关的原文片段。提取结果会在后续阶段对解决问题提供帮助。
# 请严格遵循以下规则：
# 1. 尽可能准确地提取信息：一定不要一句话总结！另外如果文本中有多个相关片段，请全部提取，而不是仅选择一个。
# 2. 保持完整性：如果某个片段部分内容与需求相关，则保留整个片段，不要只截取部分句子。
# 3. 包含相关链接：部分文本有对应的链接（可能是完整链接，或是链接标记如"url1"），如果某个链接和需求可能相关，请提取该链接及其周围相关内容，确保信息完整。如果片段中包含了引用，请同时贴上相关引用的链接；
# 4. 保持原文语言：提取的内容必须与原文语言一致，不要进行翻译或改写。
# 5. 即使没有完全对应的内容，也尽力提取可能相关的信息，这对于解决问题很有帮助。
# 6. 直接输出答案，不要有多余内容。保持准确，信息量丰富和完整，结果在1000字左右即可。

# <长文本>
# {reference}
# </长文本>

# <需求描述>
# {task}
# </需求描述>

# 提取的原文片段如下：

# """
#         input_ids = self.tokenizer(content)["input_ids"]
#         max_chunk_size = 15000
#         overlap = 500
#         num_chunks = (len(input_ids) + max_chunk_size - 1) // max_chunk_size
#         num_chunks = min(num_chunks, 5)
#         summary_parts = []

#         for i in range(num_chunks):
#             start_idx = max(0, i * max_chunk_size - overlap)
#             end_idx = min((i + 1) * max_chunk_size + overlap, len(input_ids))
#             chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])
#             prompt = final_input.format(
#                 task=description, reference=chunk
#             )
#             messages = [{"role": "user", "content": prompt}]
#             response = self.model(messages)
#             response = response.split("</think>", 1)[-1].strip()
#             if len(response) > 0:
#                 summary_parts.append(response)
#         if len(summary_parts) == 1:
#             response = summary_parts[0]
#         else:
#             response = ""
#             for i, summary_part in enumerate(summary_parts):
#                 response += f"### 第{i+1}片段抽取结果：\n{summary_part}\n"
#         return response

#     def get_url_content_from_mcp(self, url):
#         arguments = {
#             "url": url,
#             "type": "全文",
#             "skip_cache": False,
#             "from_mcp_call": True,
#             "is_offline": True,
#         }
#         arguments_str = json.dumps(arguments)
#         data = {
#             "name": "LinkReader",
#             "arguments": arguments_str,
#             "traffic_group": "NLP_LLM",
#             "traffic_id": "rlhf",
#         }
#         for _ in range(3):
#             try:
#                 resp = requests.post(
#                     "https://bytemcp.bytedance.net/plugin/search/tools/call",
#                     headers={
#                         "X-Tt-Env": "ppe_deep_research",
#                         "X-Use-Ppe": "1",
#                         "Content-Type": "application/json",
#                     },
#                     json = data
#                 )
#                 logid = resp.headers.get("x-tt-logid")
#                 resp.raise_for_status()
#                 result = resp.json()
#                 result = json.loads(result["result"])
#                 if "data" in result and "content" in result["data"]:
#                     return result["data"]["content"], logid
#                 elif "pdf_content" in result:
#                     return result["pdf_content"], logid
#                 else:
#                     return "", logid
#             except:
#                 time.sleep(1)
#         return "", ""


#     def get_url_content(self, url):
#         thought = {
#             "action": "browser_navigate",
#             "url": url,
#             "need_element_analyzer": False,
#         }

#         body = {
#             "PluginThoughtList": [
#                 {
#                     "PluginName": "",
#                     "Thought": json.dumps(thought, ensure_ascii=False).encode("utf8").decode(),
#                     "ToolName": "BrowserUseAgent",
#                     "PluginId": "100072706",
#                 }
#             ],
#             "BizId": "seed",
#             "TrafficGroup": "doubao",
#             "TrafficId": "deep_research",
#             "ak": "cZ33F9UtXDvv3qBFU1cusuRmhbz51xnZ",
#         }
#         headers = {"Content-Type": "application/json;charset=UTF-8", "x-tt-env": "ppe_20250311210028", "x-use-ppe": "1"}
#         full_text = ""
#         logid = None
#         url_dict = {}
#         for _ in range(3):
#             try:
#                 resp = requests.post("https://ah3yte2a.fn.bytedance.net/api/v1/observe?", json=body, headers=headers).json()
#                 logid = resp["data"]["plugin_request"]["Base"]["LogID"]
#                 obs_list = resp.get("data", {}).get("plugin_observation_list", [])
#                 observation = json.loads(obs_list[0]["Observation"])
#                 web_page = observation["web_page_info_list"][0]
#                 content_block_list = web_page["content_info"]["content_block_list"]
#                 if content_block_list is not None and len(content_block_list) > 0:
#                     full_text = ""
#                     for text_piece in content_block_list:
#                         if len(text_piece["link"]) == 0:
#                             full_text += text_piece["text"]
#                         elif len(text_piece["text"]) > 0:
#                             k = f"(url{len(url_dict)+1})"
#                             url_dict[k] = text_piece["link"]
#                             if self.link_mask_url:
#                                 full_text += "{}".format(text_piece["text"])
#                             else:
#                                 full_text += "{}{}".format(text_piece["text"], k)
#                 else:
#                     full_text = web_page["content_info"]["content"]
#                 break
#             except:
#                 full_text = ""
#                 url_dict = {}
#                 time.sleep(3)
#         return full_text, url_dict, logid
    
#     def get_single_url_content(self, result: dict, description: str, url: str):
#         begin_time = time.time()
#         if self.use_mcp:
#             full_text, logid = self.get_url_content_from_mcp(url)
#             url_dict = {}
#         else:
#             full_text, url_dict, logid = self.get_url_content(url)
#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             try:
#                 input_ids = self.tokenizer(full_text)["input_ids"]
#                 metric["link_reader_raw_content_len"] = len(input_ids)
#                 if len(input_ids) < self.min_summary_length:
#                     content = f"""# Page url: {url}\n# Page Summary:\n{full_text}\n"""
#                     metric["link_reader_summary"] = False
#                 else:
#                     response = self.summary_reference(full_text, description)
#                     for k, v in url_dict.items():
#                         if k in response:
#                             response = response.replace(k, "({})".format(v))
#                     rsp_ids = self.tokenizer(response)["input_ids"]
#                     if len(rsp_ids) > 3000:
#                         response = self.tokenizer.decode(rsp_ids[:3000]) + "..."
#                     content = f"""# Page url: {url}\n# Page Summary:\n{response}\n"""
#                     metric["link_reader_summary"] = True
#                 metric["browser_navigate_status"] = "success"
#             except Exception:
#                 content = f"Failed to browse content in {url}."
#                 metric["browser_navigate_status"] = "failed"
#                 metric["fialed_stage"] = "summary"
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_summary_time"] = end_time - browser_navigate_end_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])
#         result['content'] = content
#         result['metric'] = metric

#     def __call__(self, description: str, url: list):
#         begin_time = time.time()
#         metric = {"sub_metrics": []}
#         link_reader_output = ""
#         for page in url:
#             result = {}
#             thread = threading.Thread(target=self.get_single_url_content, args=(result, description, page))
#             thread.start()
#             thread.join()
#             link_reader_output += result["content"]
#             metric['sub_metrics'].append(result["metric"])
#         end_time = time.time()
#         metric["multi_link_reader_total_time"] = end_time - begin_time
#         final_result = {
#             "content": link_reader_output,
#             "metric": metric,
#         }
#         return final_result


# class LinkReaderPluginToolv3RL:
#     name = "LinkReader"
#     description = """A browsing tool that will parse the content from a given url(web page or pdf). The url should be the existing urls obtained from the previous search and browse results, rather than fabricating it. Perform url and description about what your requirement, you will get two things about the page, 1. a detailed summary 2. the raw html page.
# The calling example is as follow:
# <action>
# {
#     "name": "LinkReader",
#     "arguments": {
#         "description": "your requirement of this url",
#         "url": "target url"
#     },
# }
# </action>
# LinkReader can be used in three ways:
# 1. Directly: If the task already provides a URL, you can directly call LinkReader to extract content from that URL.
# 2. After Search: If a search result (from the Search tool) provides a URL with a snippet, you can follow up with LinkReader to extract more detailed information from the page.
# 3. Following Extracted Links: If a previous LinkReader call returns content containing URLs, and those links may contain useful information, you can extract and process them using LinkReader.
# • If the link is a full URL (starting with http), it can be used directly.
# • If the link is a relative path (e.g., /news, #cite_note-oni-5), it should be appended to the current URL or the site’s base URL to form a complete URL before calling LinkReader again.
# This allows for deeper content extraction and ensures more detailed information retrieval.
# """
#     def __init__(self, min_summary_length=5000, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         if model_type == "deepseek-r1":
#             self.model = LLMModelClient(model_type="deepseek-r1")
#         else:
#             self.model = LLMModelClient(model_type="doubao", model_psm=model_psm, model_cluster=model_cluster, temperature=0.01)
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode", temperature=0.01)  # 15b
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.sed.webgpt_100b_debug", model_cluster="default", temperature=0.01)  # 2b5
#         self.min_summary_length = min_summary_length

#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         if os.path.exists(tokenizer_path):
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         else:
#             self.tokenizer = None
    
#     def summary_reference(self, content, description):
#         final_input = """你是一名信息收集专家，你的团队正在合作解决一个很难的问题，你的任务是从提供的 <长文本>（可能是文章或网页内容）中，尽可能准确地提取与 <需求描述> 相关的原文片段，包括文本和可能相关的网址。提取结果会在后续阶段对解决问题提供帮助。
# 请严格遵循以下规则：
# 1. 有用信息既包括相关文本内容，也包括任何可能包含其他相关信息的网址链接。
# 2. 尽可能准确地提取信息：一定不要一句话总结！另外如果文本中有多个相关片段，请全部提取，而不是仅选择一个。
# 3. 仅提取原始文本片段：直接从文本中输出原始相关内容，不要总结、概括或改写。
# 4. 保持完整性：如果某个片段部分内容与需求相关，则保留整个片段，不要只截取部分句子。
# 5. 包含相关链接：部分文本有对应的链接（可能是完整链接，或是链接标记如"url1"），如果某个链接和需求可能相关，请提取该链接及其周围相关内容，确保信息完整。如果片段中包含了引用，请同时贴上相关引用的链接；
# 6. 网址链接统一处理：请注意，某些网址链接可能指向包含潜在信息的网站，后续阶段会决定是否进一步探索它们。因此，请包括所有可能有用的网址链接及其说明。
# 7. 保持原文语言：提取的内容必须与原文语言一致，不要进行翻译或改写。
# 8. 即使没有完全对应的内容，也尽力提取可能相关的信息，这对于解决问题很有帮助。
# 9. 直接输出答案，不要有多余内容。保持准确，信息量丰富和完整，结果在1000字左右即可。

# <长文本>
# {reference}
# </长文本>

# <需求描述>
# {task}
# </需求描述>

# 提取的原文片段如下：

# """
#         input_ids = self.tokenizer(content)["input_ids"]
#         max_chunk_size = 15000
#         overlap = 500
#         num_chunks = (len(input_ids) + max_chunk_size - 1) // max_chunk_size
#         num_chunks = min(num_chunks, 5)
#         summary_parts = []

#         for i in range(num_chunks):
#             start_idx = max(0, i * max_chunk_size - overlap)
#             end_idx = min((i + 1) * max_chunk_size + overlap, len(input_ids))
#             chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])
#             prompt = final_input.format(
#                 task=description, reference=chunk
#             )
#             messages = [{"role": "user", "content": prompt}]
#             response = self.model(messages)
#             response = response.split("</think>", 1)[-1].strip()
#             if len(response) > 0:
#                 summary_parts.append(response)
#         if len(summary_parts) == 1:
#             response = summary_parts[0]
#         else:
#             response = ""
#             for i, summary_part in enumerate(summary_parts):
#                 response += f"### 第{i+1}片段抽取结果：\n{summary_part}\n"
#         return response

#     def get_url_content(self, url):
#         thought = {
#             "action": "browser_navigate",
#             "url": url,
#             "need_element_analyzer": False,
#         }
#         body = {
#             "PluginThoughtList": [
#                 {
#                     "PluginName": "",
#                     "Thought": json.dumps(thought, ensure_ascii=False).encode("utf8").decode(),
#                     "ToolName": "BrowserUseAgent",
#                     "PluginId": "100072706",
#                 }
#             ],
#             "BizId": "seed",
#             "TrafficGroup": "doubao",
#             "TrafficId": "browsecomp",
#             "ak": "cZ33F9UtXDvv3qBFU1cusuRmhbz51xnZ",
#         }
#         log_id = logid.generate()
#         headers = {"Content-Type": "application/json;charset=UTF-8", "x-tt-env": "ppe_20250311210028", "x-use-ppe": "1", "x-tt-logid": log_id}
#         observation = None
#         # logid = None
#         resp = None
#         for _ in range(3):
#             try:
#                 resp = requests.post("https://ah3yte2a.fn.bytedance.net/api/v1/observe?", json=body, headers=headers, timeout=500).json()
#                 # logid = resp["data"]["plugin_request"]["Base"]["LogID"]
#                 obs_list = resp.get("data", {}).get("plugin_observation_list", [])
#                 observation = json.loads(obs_list[0]["Observation"])
#                 web_page = observation["web_page_info_list"][0]
#                 full_text = ""
#                 if web_page["content_info"]["content_block_list"] is not None and len(web_page["content_info"]["content_block_list"]) > 0:
#                     for text_piece in web_page["content_info"]["content_block_list"]:
#                         full_text += text_piece["text"]
#                 else:
#                     full_text = web_page["content_info"]["content"]
#                 assert len(full_text) > 0
#                 break
#             except Exception as e:
#                 print(f'linkreaderv3 failed url {url}')
#                 print(f'linkreaderv3 failed resp {resp}')
#                 print(f'linkreaderv3 failed logid {log_id}')
#                 # traceback.print_exc()
#                 # print(f'get url content failed {e}')
#                 observation = None
#         url_dict = {}
#         content_block_list = []
#         if observation["web_page_info_list"][0]["content_info"].get("link_block_list", None) is not None:
#             content_block_list += observation["web_page_info_list"][0]["content_info"]["link_block_list"]
#         if  observation["web_page_info_list"][0]["content_info"].get("content_block_list", None) is not None:
#             content_block_list += observation["web_page_info_list"][0]["content_info"]["content_block_list"]
#         if len(content_block_list) > 0:
#             full_text = ""
#             for text_piece in content_block_list:
#                 if len(text_piece["link"]) == 0:
#                     full_text += text_piece["text"]
#                 elif len(text_piece["text"]) > 0:
#                     k = f"url{len(url_dict)+1}"
#                     url_dict[k] = text_piece["link"]
#                     full_text += "{}({})".format(text_piece["text"], k)
#         else:
#             full_text = observation["web_page_info_list"][0]["content_info"]["content"]
#         full_text_real_url = copy.deepcopy(full_text)
#         for k, v in url_dict.items():
#             if k in full_text_real_url:
#                 full_text_real_url = full_text_real_url.replace(k, "{}".format(v))
#         return full_text_real_url, url_dict, log_id
        
#     def __call__(self, description: str, url: str) -> str:
#         begin_time = time.time()
#         full_text, url_dict, logid = self.get_url_content(url)
#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             try:
#                 input_ids = self.tokenizer(full_text)["input_ids"]
#                 metric["link_reader_raw_content_len"] = len(input_ids)
#                 if len(input_ids) < self.min_summary_length:
#                     content = f"""# Page url: {url}\n# Page Summary:\n{full_text}\n"""
#                     metric["link_reader_summary"] = False
#                 else:
#                     response = self.summary_reference(full_text, description)
#                     for k, v in url_dict.items():
#                         if k in response:
#                             response = response.replace(k, "({})".format(v))
#                     rsp_ids = self.tokenizer(response)["input_ids"]
#                     if len(rsp_ids) > 3000:
#                         response = self.tokenizer.decode(rsp_ids[:3000]) + "..."
#                     content = f"""# Page url: {url}\n# Page Summary:\n{response}\n"""
#                     metric["link_reader_summary"] = True
#                 metric["browser_navigate_status"] = "success"
#             except Exception:
#                 content = f"Failed to browse content in {url}."
#                 metric["browser_navigate_status"] = "failed"
#                 metric["fialed_stage"] = "summary"
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_summary_time"] = end_time - browser_navigate_end_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])
#         result = {
#             "content": content,
#             "metric": metric,
#         }
#         return result


# class MultiLinkReaderPluginToolv3RL:
#     name = "MultiLinkReader"
#     description = """A browsing tool that will parse the content from a given url(web page or pdf). The url should be the existing urls obtained from the previous search and browse results, rather than fabricating it. Perform url and description about what your requirement, you will get two things about the page, 1. a detailed summary 2. the raw html page.
# The calling example is as follow:
# <action>
# {
#     "name": "LinkReader",
#     "arguments": {
#         "description": "your requirement of this url",
#         "url": "target url"
#     },
# }
# </action>
# LinkReader can be used in three ways:
# 1. Directly: If the task already provides a URL, you can directly call LinkReader to extract content from that URL.
# 2. After Search: If a search result (from the Search tool) provides a URL with a snippet, you can follow up with LinkReader to extract more detailed information from the page.
# 3. Following Extracted Links: If a previous LinkReader call returns content containing URLs, and those links may contain useful information, you can extract and process them using LinkReader.
# • If the link is a full URL (starting with http), it can be used directly.
# • If the link is a relative path (e.g., /news, #cite_note-oni-5), it should be appended to the current URL or the site’s base URL to form a complete URL before calling LinkReader again.
# This allows for deeper content extraction and ensures more detailed information retrieval.
# """
#     def __init__(self, min_summary_length=5000, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         if model_type == "deepseek-r1":
#             self.model = LLMModelClient(model_type="deepseek-r1")
#         else:
#             self.model = LLMModelClient(model_type="doubao", model_psm=model_psm, model_cluster=model_cluster, temperature=0.01)
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode", temperature=0.01)  # 15b
#             # self.model = LLMModelClient(model_type="doubao", model_psm="search.sed.webgpt_100b_debug", model_cluster="default", temperature=0.01)  # 2b5
#         self.min_summary_length = min_summary_length

#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         if os.path.exists(tokenizer_path):
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         else:
#             self.tokenizer = None
    
#     def summary_reference(self, content, description):
#         final_input = """你是一名信息收集专家，你的团队正在合作解决一个很难的问题，你的任务是从提供的 <长文本>（可能是文章或网页内容）中，尽可能准确地提取与 <需求描述> 相关的原文片段，包括文本和可能相关的网址。提取结果会在后续阶段对解决问题提供帮助。
# 请严格遵循以下规则：
# 1. 有用信息既包括相关文本内容，也包括任何可能包含其他相关信息的网址链接。
# 2. 尽可能准确地提取信息：一定不要一句话总结！另外如果文本中有多个相关片段，请全部提取，而不是仅选择一个。
# 3. 仅提取原始文本片段：直接从文本中输出原始相关内容，不要总结、概括或改写。
# 4. 保持完整性：如果某个片段部分内容与需求相关，则保留整个片段，不要只截取部分句子。
# 5. 包含相关链接：部分文本有对应的链接（可能是完整链接，或是链接标记如"url1"），如果某个链接和需求可能相关，请提取该链接及其周围相关内容，确保信息完整。如果片段中包含了引用，请同时贴上相关引用的链接；
# 6. 网址链接统一处理：请注意，某些网址链接可能指向包含潜在信息的网站，后续阶段会决定是否进一步探索它们。因此，请包括所有可能有用的网址链接及其说明。
# 7. 保持原文语言：提取的内容必须与原文语言一致，不要进行翻译或改写。
# 8. 即使没有完全对应的内容，也尽力提取可能相关的信息，这对于解决问题很有帮助。
# 9. 直接输出答案，不要有多余内容。保持准确，信息量丰富和完整，结果在1000字左右即可。

# <长文本>
# {reference}
# </长文本>

# <需求描述>
# {task}
# </需求描述>

# 提取的原文片段如下：

# """
#         input_ids = self.tokenizer(content)["input_ids"]
#         max_chunk_size = 15000
#         overlap = 500
#         num_chunks = (len(input_ids) + max_chunk_size - 1) // max_chunk_size
#         num_chunks = min(num_chunks, 5)
#         summary_parts = []

#         for i in range(num_chunks):
#             start_idx = max(0, i * max_chunk_size - overlap)
#             end_idx = min((i + 1) * max_chunk_size + overlap, len(input_ids))
#             chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])
#             prompt = final_input.format(
#                 task=description, reference=chunk
#             )
#             messages = [{"role": "user", "content": prompt}]
#             response = self.model(messages)
#             response = response.split("</think>", 1)[-1].strip()
#             if len(response) > 0:
#                 summary_parts.append(response)
#         if len(summary_parts) == 1:
#             response = summary_parts[0]
#         else:
#             response = ""
#             for i, summary_part in enumerate(summary_parts):
#                 response += f"### 第{i+1}片段抽取结果：\n{summary_part}\n"
#         return response

#     def get_url_content(self, url):
#         thought = {
#             "action": "browser_navigate",
#             "url": url,
#             "need_element_analyzer": False,
#         }
#         body = {
#             "PluginThoughtList": [
#                 {
#                     "PluginName": "",
#                     "Thought": json.dumps(thought, ensure_ascii=False).encode("utf8").decode(),
#                     "ToolName": "BrowserUseAgent",
#                     "PluginId": "100072706",
#                 }
#             ],
#             "BizId": "seed",
#             "TrafficGroup": "doubao",
#             "TrafficId": "browsecomp",
#             "ak": "cZ33F9UtXDvv3qBFU1cusuRmhbz51xnZ",
#         }
#         log_id = logid.generate()
#         headers = {"Content-Type": "application/json;charset=UTF-8", "x-tt-env": "ppe_20250311210028", "x-use-ppe": "1", "x-tt-logid": log_id}
#         observation = None
#         # logid = None
#         resp = None
#         for _ in range(3):
#             try:
#                 resp = requests.post("https://ah3yte2a.fn.bytedance.net/api/v1/observe?", json=body, headers=headers, timeout=500).json()
#                 # logid = resp["data"]["plugin_request"]["Base"]["LogID"]
#                 obs_list = resp.get("data", {}).get("plugin_observation_list", [])
#                 observation = json.loads(obs_list[0]["Observation"])
#                 web_page = observation["web_page_info_list"][0]
#                 full_text = ""
#                 if web_page["content_info"]["content_block_list"] is not None and len(web_page["content_info"]["content_block_list"]) > 0:
#                     for text_piece in web_page["content_info"]["content_block_list"]:
#                         full_text += text_piece["text"]
#                 else:
#                     full_text = web_page["content_info"]["content"]
#                 assert len(full_text) > 0
#                 break
#             except Exception as e:
#                 print(f'linkreaderv3 failed url {url}')
#                 print(f'linkreaderv3 failed resp {resp}')
#                 print(f'linkreaderv3 failed logid {log_id}')
#                 # traceback.print_exc()
#                 # print(f'get url content failed {e}')
#                 observation = None
#         url_dict = {}
#         content_block_list = []
#         if observation["web_page_info_list"][0]["content_info"].get("link_block_list", None) is not None:
#             content_block_list += observation["web_page_info_list"][0]["content_info"]["link_block_list"]
#         if  observation["web_page_info_list"][0]["content_info"].get("content_block_list", None) is not None:
#             content_block_list += observation["web_page_info_list"][0]["content_info"]["content_block_list"]
#         if len(content_block_list) > 0:
#             full_text = ""
#             for text_piece in content_block_list:
#                 if len(text_piece["link"]) == 0:
#                     full_text += text_piece["text"]
#                 elif len(text_piece["text"]) > 0:
#                     k = f"url{len(url_dict)+1}"
#                     url_dict[k] = text_piece["link"]
#                     full_text += "{}({})".format(text_piece["text"], k)
#         else:
#             full_text = observation["web_page_info_list"][0]["content_info"]["content"]
#         full_text_real_url = copy.deepcopy(full_text)
#         for k, v in url_dict.items():
#             if k in full_text_real_url:
#                 full_text_real_url = full_text_real_url.replace(k, "{}".format(v))
#         return full_text_real_url, url_dict, log_id
        
#     def get_single_url_content(self, result: dict, description: str, url: str) -> str:
#         begin_time = time.time()
#         full_text, url_dict, logid = self.get_url_content(url)
#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             try:
#                 input_ids = self.tokenizer(full_text)["input_ids"]
#                 metric["link_reader_raw_content_len"] = len(input_ids)
#                 if len(input_ids) < self.min_summary_length:
#                     content = f"""# Page url: {url}\n# Page Summary:\n{full_text}\n"""
#                     metric["link_reader_summary"] = False
#                 else:
#                     response = self.summary_reference(full_text, description)
#                     for k, v in url_dict.items():
#                         if k in response:
#                             response = response.replace(k, "({})".format(v))
#                     rsp_ids = self.tokenizer(response)["input_ids"]
#                     if len(rsp_ids) > 3000:
#                         response = self.tokenizer.decode(rsp_ids[:3000]) + "..."
#                     content = f"""# Page url: {url}\n# Page Summary:\n{response}\n"""
#                     metric["link_reader_summary"] = True
#                 metric["browser_navigate_status"] = "success"
#             except Exception:
#                 content = f"Failed to browse content in {url}."
#                 metric["browser_navigate_status"] = "failed"
#                 metric["fialed_stage"] = "summary"
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_summary_time"] = end_time - browser_navigate_end_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])
#         result['content'] = content
#         result['metric'] = metric
    
#     def __call__(self, description: str, url: list):
#         begin_time = time.time()
#         metric = {"sub_metrics": []}
#         link_reader_output = ""
#         for page in url:
#             result = {}
#             thread = threading.Thread(target=self.get_single_url_content, args=(result, description, page))
#             thread.start()
#             thread.join()
#             link_reader_output += result["content"]
#             metric['sub_metrics'].append(result["metric"])
#         end_time = time.time()
#         metric["multi_link_reader_total_time"] = end_time - begin_time
#         final_result = {
#             "content": link_reader_output,
#             "metric": metric,
#         }
#         return final_result


# class TextBrowserView:
#     name = "TextBrowserView"
#     description = """A browsing tool that will parse the content from a given url(web page or pdf). The url should be the existing urls obtained from the previous search and browse results, rather than fabricating it. Perform url and description about what your requirement, you will get two things about the page, 1. a detailed summary 2. the raw html page.
# The calling example is as follow:
# <action>
# {
#     "name": "TextBrowserView",
#     "arguments": {
#         "description": "your requirement of this url",
#         "url": "target url"
#     },
# }
# </action>
# TextBrowserView can be used in three ways:
# 1. Directly: If the task already provides a URL, you can directly call TextBrowserView to extract content from that URL.
# 2. After Search: If a search result (from the Search tool) provides a URL with a snippet, you can follow up with TextBrowserView to extract more detailed information from the page.
# 3. Following Extracted Links: If a previous TextBrowserView call returns content containing URLs, and those links may contain useful information, you can extract and process them using TextBrowserView.
# • If the link is a full URL (starting with http), it can be used directly.
# • If the link is a relative path (e.g., /news, #cite_note-oni-5), it should be appended to the current URL or the site’s base URL to form a complete URL before calling TextBrowserView again.
# This allows for deeper content extraction and ensures more detailed information retrieval.
# """
#     def __init__(self, min_summary_length=10000, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         if os.path.exists(tokenizer_path):
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         else:
#             self.tokenizer = None

#     def get_url_content(self, url, description, abparams={}):
#         fixparams = {"is_offline":True, "url": url, "need_image_content": False, "description": description, "summarize_config": {"enable_summarize": True}}
#         fixparams.update(abparams)
#         result = mcp_client.tools_call("TextBrowserView", fixparams)
#         full_text = []
#         try:
#             valid_content = result['result']['documents'][0]['content']
#             for i in range(len(valid_content)):
#                 if valid_content[i]['type'] == 'text':
#                     full_text.append(valid_content[i]['text'])
#         except:
#             pass
#         logid = result['log_id']
#         full_text = '\n'.join(full_text)
#         return full_text, logid
        
#     def __call__(self, description: str, url: str, page_idx: int, abparams={}) -> str:
#         begin_time = time.time()
#         full_text, logid = self.get_url_content(url, description, abparams)
#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             content = "<|superscript|>:"+str(page_idx+1)+"：\n"+full_text
#             metric["browser_navigate_status"] = "success"
#             page_idx += 1
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])
    
#         result = {
#             "content": content[:30000],
#             "metric": metric,
#             "page_idx": page_idx
#         }
#         return result


# class MultiTextBrowser:
#     name = "MultiTextBrowser"
#     description = ""
#     def __init__(self, min_summary_length=10000, model_type="deepseek-r1", model_psm="search.nlp.dev_ww_1", model_cluster="m2n_decode"):
#         super().__init__()
#         tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
#         if os.path.exists(tokenizer_path):
#             self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         else:
#             self.tokenizer = None
    
#     def get_url_content(self, url, description):
#         fixparams = {"is_offline":True, "url": url, "need_image_content": False, "description": description, "summarize_config": {"enable_summarize": True}}
#         result = mcp_client.tools_call("TextBrowserView", fixparams)
#         full_text = []
#         try:
#             valid_content = result['result']['documents'][0]['content']
#             for i in range(len(valid_content)):
#                 if valid_content[i]['type'] == 'text':
#                     full_text.append(valid_content[i]['text'])
#         except:
#             pass
#         logid = result['log_id']
#         full_text = '\n'.join(full_text)
#         return full_text, logid
        
#     def get_single_url_content(self, result: dict, description: str, url: str, page_idx: int) -> str:
#         begin_time = time.time()
#         full_text, logid = self.get_url_content(url, description)
#         browser_navigate_end_time = time.time()
#         browser_navigate_time = browser_navigate_end_time - begin_time
#         metric = {
#             "browser_navigate_time": browser_navigate_time,
#             "logid": logid,
#         }
#         if len(full_text) == 0:
#             content = f"Failed to browse content in {url}."
#             metric["browser_navigate_status"] = "failed"
#             metric["fialed_stage"] = "browse"
#         else:
#             content = "<|superscript|>:"+str(page_idx+1)+"：\n"+full_text
#             metric["browser_navigate_status"] = "success"
#             page_idx += 1
#         end_time = time.time()
#         metric["link_reader_total_time"] = end_time - begin_time
#         metric["link_reader_total_len"] = len(self.tokenizer(content)["input_ids"])

#         result['content'] = content[:30000]
#         result['metric'] = metric
#         result['page_idx'] = page_idx
    
#     def __call__(self, description: str, urls: list, page_idx: int):
#         begin_time = time.time()
#         metric = {"sub_metrics": []}
#         link_reader_output = ""
#         for page in urls:
#             result = {}
#             thread = threading.Thread(target=self.get_single_url_content, args=(result, description, page, page_idx))
#             thread.start()
#             thread.join()
#             link_reader_output += result["content"]
#             metric['sub_metrics'].append(result["metric"])
#             page_idx = result['page_idx']
#         end_time = time.time()
#         metric["multi_link_reader_total_time"] = end_time - begin_time
#         final_result = {
#             "content": link_reader_output,
#             "metric": metric,
#             "page_idx": page_idx
#         }
#         return final_result


class Fetch:
    name = "Fetch"
    description = ""
    def __init__(self, return_prompt_type='text', adapter=None):
        super().__init__()
        self.return_prompt_type = return_prompt_type
        self.adapter = adapter
        self.ignore_image = True
        tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_v5.1_deepresearch_0416"
        if os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None
        self.api_key = "d294932a-248c-44d4-93b6-13b65e00f3e4"
        #self.url = "https://gpt.bytedance.net/gpt/tool_hub/online/apihub/function_call_proxy"
        self.url = "https://gpt.bytedance.net/gpt/tool_hub/online/mcp_server/proxy/global_search_v2/mcp"
        self.headers = { "api-key": self.api_key, "Content-Type": "application/json" }
    
    def get_response(self, fetch_request_list: list):
        for req in fetch_request_list:
            req['snippet']['max_length'] = 3000   # 精读长度最长是3k
        #input_params = json.dumps({"fetch_request_list": fetch_request_list})
        #data = { "name": "Fetch", "input_params": input_params, "api_id": "6314" }
        data = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "Fetch",
                "arguments": {
                    "fetch_request_list": fetch_request_list
                }
            }
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code != 200:
            return None, {"status_message": f"Failed to call API Hub: {response.status_code} {response.text} {response.headers}"}
        log_id, result = response.headers.get("X-Tt-Logid"), response.json()
        #fetch_result = json.loads(result.get("data", {}).get("result", {}))
        fetch_result = json.loads(result['result']['content'][0]['text'])['result']
        return fetch_result, {"status_message": "success", "status_code": 0, "log_id": log_id}
    
    def adapt_to_text(self, fetch_result: dict, page_idx: int):
        rsp_str = ''
        for fetch_response in fetch_result['fetch_response_list']:  # 工具内部已经有截断(摘要最大长度，默认4000)，这里和线上保持一致，不做额外截断
            if fetch_response.get('status_code', -1) != 0:
                rsp_str += "<|superscript|>:{}：\nFailed to browse content in {}\n".format(page_idx+1, fetch_response.get("url", ""))
            else:
                rsp_str += "<|superscript|>:{}：\nPage url:{}\nPage Summary:{}\n".format(
                    page_idx+1, fetch_response.get("url", ""), ''.join([item.get("text", "") if item.get("type", "") == 'text' else f"<id={i}, type='link', url={item.get('link', {}).get('target_url', '')} content='{item.get('text', '')}'>" for i, item in enumerate(fetch_response.get("snippet", [])) if item.get('type', "") != 'image']).strip()
                )  # snippet是图/文/link list，图片跳过，其他直接拼接。如果工具内部没有正常解析出\n，找zhuoling提badcase
            page_idx += 1 
        return rsp_str, page_idx
    
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
    
    def _porcess_single_fetch_result_prompt(self, fetch_request, fetch_response, page_idx):
        fetch_id = fetch_response.get("fetch_id", "")
        pagination_content = fetch_response.get("content", [])
        snippet_content = fetch_response.get("snippet", [])

        root = ET.Element("fetch")
        source_reference_id = fetch_request.get("reference_id", "")
        if len(source_reference_id) > 0:
            root.set("source_reference_id", source_reference_id)
        if len(fetch_id) > 0:
            root.set("fetch_id", fetch_id)

        document = ET.SubElement(root, "document")
        
        reference_type = "<|superscript|>:"
        current_id = page_idx+1
        page_idx += 1
        
        reference_id = f"{reference_type}{current_id}"
        request_reference_id = fetch_request.get("reference_id", "")

        document.set("reference_id", reference_id)

        start_offset = fetch_response.get("page_token_offset", 0)
        end_offset = start_offset + fetch_response.get("page_token_count", 0)

        fetch_url = fetch_request.get("url", "")

        if len(fetch_url) > 0:
            url_et = ET.SubElement(document, "url")
            url_et.text = fetch_url

        fetch_title = fetch_response.get("title", "")
        if len(fetch_title) > 0:
            title_et = ET.SubElement(document, "title")
            title_et.text = fetch_title

        content_type, content = "pagination_content", pagination_content
        if len(fetch_request.get("snippet", {}).get("query", "")) > 0 and len(snippet_content) > 0:
            content_type, content = "snippet", snippet_content

        content = self._filter_content_part(content)
        if len(content) > 0:
            content_et = ET.SubElement(document, content_type)
            if content_type == "pagination_content":
                content_et.set("start_offset", str(start_offset))
                content_et.set("end_offset", str(end_offset))

            # Use the common mixed content processing function
            self._process_mixed_content(content_et, content, reference_id, "fetch")
        status_code, status_message = fetch_response.get("status_code", 0), fetch_response.get("status_message", "success")

        if status_code != 0 and len(status_message) > 0 and status_message != "success":
            text_et = ET.SubElement(document, "err_msg")
            text_et.text = status_message

        # 将XML转换为字符串并美化格式
        return self._transform_to_output(root), page_idx

    def adapt_to_xml(self, fetch_request_list: list, fetch_result: dict, page_idx: int):
        results = list()
        for fetch_request, fetch_response in zip(fetch_request_list, fetch_result['fetch_response_list']):
            content, page_idx = self._porcess_single_fetch_result_prompt(fetch_request, fetch_response, page_idx)
            results.append(content)
        return "\n".join(results), page_idx
    
    def __call__(self, fetch_request_list, page_idx=0, refid_url_map=None, no_display_url_map=None, url_tosurl_map=None, tosurl_url_map=None, url_no_display_map=None):

        if self.return_prompt_type == 'text':
            raw_fetch_request_list = copy.deepcopy(fetch_request_list)
            for fetch_request in fetch_request_list:  
                if 'url' not in fetch_request:
                    fetch_request["url"] = refid_url_map.get(fetch_request['reference_id'], "")
                if fetch_request["url"] in no_display_url_map: # 不可见url->真url
                    fetch_request["url"] = no_display_url_map[fetch_request["url"]]
                if fetch_request["url"] in url_tosurl_map:  # 替换url -> tosurl
                    fetch_request["url"] = url_tosurl_map[fetch_request["url"]]

            fetch_result, metrics = self.get_response(fetch_request_list)

            for fetch_response in fetch_result['fetch_response_list']:
                if fetch_response.get("url", "") in tosurl_url_map:
                    fetch_response["url"] = tosurl_url_map[fetch_response.get("url", "")]
                if fetch_response.get("url", "") in url_no_display_map:
                    fetch_response["url"] = url_no_display_map[fetch_response.get("url", "")]

            rsp_str, page_idx = self.adapt_to_text(fetch_result, page_idx)
        else:
            fetch_result, metrics = self.get_response(fetch_request_list)
            rsp_str, page_idx = self.adapt_to_xml(fetch_request_list, fetch_result, page_idx)
            # reformed_fetch_request_list = self.adapter.reformat_request_fetch({"fetch_request_list": fetch_request_list})
            # fetch_result, metrics = self.get_response(reformed_fetch_request_list["fetch_request_list"])
            # tool_prompt, _ = self.adapter.get_fetch_result_prompt_and_refids(reformed_fetch_request_list, fetch_result)
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
    #linkreader = LinkReaderPluginTool(use_mcp=True)
    #linkreader = LinkReaderPluginToolV4(use_mcp=True)
    # adapter = SeedApplicationAiSearchPromptAdapter(ignore_image=True)
    fetch_tool = Fetch(return_prompt_type='xml')
    # start_time = time.time()
    # description = "武汉和九省通衢的关系"
    # url = "https://zh.wikipedia.org/wiki/%E6%AD%A6%E6%B1%89%E5%B8%82"
    # url = "https://mp.weixin.qq.com/s/Cplhv1BeWc_B5hGQX2tzKA"
    # result = linkreader(description, url)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    # end_time = time.time()
    # print("time cost: {:.2f} seconds.".format(end_time - start_time))

    # res, logid = linkreader.get_url_content_from_mcp("https://xueqiu.com/8218623339/329658335")
    # print(res[:1000])
    # print("logid: ", logid)
    arguments = {"fetch_request_list":[{"url":"http://www.cqvip.com/QK/94183X/201906/7001465878.html?sign=b99438d1fc178589c37cc23daf8f9548df0b1572c4078798f1e6f861ba8a2d37&expireTime=1817049600000&resourceId=bytedance","snippet":{"query":"总结一下"}}]}
    result = fetch_tool(fetch_request_list=arguments['fetch_request_list'], page_idx=0, refid_url_map={}, no_display_url_map={}, url_tosurl_map={}, tosurl_url_map={}, url_no_display_map={})
    print(result['content'])


if __name__ == "__main__":
    uni_test()