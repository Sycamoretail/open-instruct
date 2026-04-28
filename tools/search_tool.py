import json
import requests
import time
import datetime
from transformers import AutoTokenizer
import dateutil
import xml.etree.ElementTree as ET
# from byted_doubaoagent_adapter import SeedApplicationAiSearchPromptAdapter

# tokenizer_path = "/mlx_devbox/users/luoyunze/playground/deep_research/tiny_alignment_data/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0604"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# class SearchPluginTool:
#     name = "Search"
#     description = """工具"搜索"（Search）介绍：
# - 功能说明：这是一个联网搜索工具，输入仅1个搜索问题，返回多个搜索结果，每个搜索结果以“搜索结果+序号”开始，包括标题、网页主要内容、网页发布时间等信息。
# - 使用场景：当你需要获取更多信息时，特别是目前已知信息存在如下问题时（1）关键信息缺失/矛盾，（2）数据时效性不足（3），存在更权威信源可能性，（4）可丰富回答维度，应该调用本工具进行信息的获取。请注意，本搜索工具是面向通用场景，如果想要搜索垂直领域，且正好有对应领域的垂直搜索工具，可以优先调用垂直搜索工具获得更好的结果，在还需要补充信息的情况下再调用本搜索工具进行补充。
# 要求：
# 1. 一次使用"搜索"的搜索词只能有一个。
# 2. 对于搜索，灵活判断使用中文还是英文的搜索词，因为query词可以使用中文和英文，对于一些可能在外文网站上的信息，优先使用英文搜索，对于其余信息则可以优先使用中文搜索。整体确保搜索query词长度不要过长（英文query词不限长度），query词不和已搜索过的query重复。
#  调用格式示例：
# <action>
# {
#     "name": "Search",
#     "arguments": {
#         "queries": ["BMW's equity structure 2025"]
#     },
# }
# </action>
# """

#     def __init__(self, use_mcp=True, use_abstract=True):
#         super().__init__()
#         self.use_mcp = use_mcp
#         self.use_abstract = use_abstract

#     def __call__(self, queries, page_idx, doc_ids=[], gids=[], max_total_len=16000):
#         abparams = {
#             "search": {
#                 "enable_deep_research_url_handle": True,
#                 "seed_plugin_summary_english_joint_i18n": True,
#                 "seed_plugin_summary_topk": 7,
#                 "seed_plugin_summary_global_search_topk": 7,
#                 "seed_plugin_summary_query_to_n_doc_rerank": 7,
#                 "seed_plugin_summary_fc_add_date_time": False,
#                 "seed_plugin_summary_fc_repeat_query": False,
#                 "browsing_add_extra_video_search": False,
#                 "enable_query_importance": False,
#                 "global_search_use_full_content": False,
#                 "enable_global_search_doubao_finance_stream": True,
#                 "enable_prompt_embed_local": True,
#                 "enable_doubao_universal_intents": True,
#                 "text2sql_char_limit": 3000,
#                 "text2sql_table_compress": True,
#                 "text2sql_add_origin_sql_v2": True,
#                 "doubao_global_search": {
#                     "search": {
#                         "doubao_enable_pdf": 1,
#                         "text2sql_model_desc_name": "18108",
#                         "open_small_doubao_text2sql": 1,
#                         "enable_text_2_sql_recall": 1,
#                         "new_engine_trace": 1,
#                         "aladdin_recall_timeout_ms": 1500,
#                         "aladdin_trace_timeout": 1300
#                     }
#                 },
#                 "doubao_vertical": {
#                     "search": {
#                         "monad": {
#                             "req_index_name_list": [
#                                 "doubao_finance",
#                                 "doubao_finance_pdf"
#                                 ],
#                             "doubao_vertical": {
#                                 "req_index_name_list": [
#                                     "doubao_finance",
#                                     "doubao_finance_pdf"
#                                 ]
#                             }
#                         }
#                     }
#                 }
#             },
#         }

#         if self.use_abstract:
#             abparams["search"].update(
#                 {
#                     "global_search_use_full_content": True,
#                     "enable_doubao_summary_extract": True,
#                     "seed_plugin_max_text_doc_num": 50,
#                     "seed_plugin_max_video_doc_num": 50,
#                     "seed_plugin_global_search_doubao_summary_extract_mode": 1,
#                     "seed_plugin_global_search_summary_extract_mode": 1,
#                     "seed_plugin_douyin_summary_extract_mode": 1,
#                     "seed_plugin_summary_extract_global_search_doubao_doc_max_str_len": 200,
#                     "seed_plugin_summary_extract_global_search_doc_max_str_len": 500,
#                     "seed_plugin_summary_extract_douyin_doc_max_str_len": 200,
#                     "seed_plugin_summary_extract_global_search_doubao_threshold": 1.2,
#                     "seed_plugin_summary_extract_global_search_threshold": 1.2,
#                     "seed_plugin_summary_extract_douyin_threshold": 1.2,
#                     "global_search": {
#                         "search": {
#                             "summary_model_downstream": True,
#                         }
#                     }
#                 }
#             )
#             abparams["search"]["doubao_global_search"]["search"]["summary_model_downstream"] = True

#         arguments = {
#             "input_query": queries[0],
#             "query": queries,
#             "model_config": {
#                 "final_prompt_version": "fc-v1",
#                 "max_total_len": max_total_len,
#                 "tokenizer_name": "bbpe155k-add_webgpt_fc"
#             },
#             "SearchCommonInfo": {
#                 "appId": "497858",
#                 "localeInfo": {
#                     "city": "北京",
#                     "district": "海淀"
#                 },
#                 "abParams": json.dumps(abparams),
#             },
#             "from_mcp_call": True,
#             "fc_search_ctrl": {"start_ref_idx": page_idx}
#         }
#         arguments_str = json.dumps(arguments)
#         data = {
#             "name": "Search",
#             "arguments": arguments_str,
#             "traffic_group": "NLP_LLM",
#             "traffic_id": "rlhf",
#         }

#         logid = None
#         content = ""
#         references = []
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
#                 content = result["content"]
#                 references = result["references"]
#                 break
#             except:
#                 time.sleep(2)

#         result = {
#             "content": content,
#             "gids": [],
#             "doc_ids": [],
#             "metric": {"logid": logid},
#             "page_idx": page_idx + len(references),
#             "content_tokens": len(tokenizer.encode(content))
#         }
#         return result


# class SearchPluginToolV2:
#     name = "Search"
#     description = """A web search tool like google, give it a list of search queries and return several related pages. The calling example is as follow:
# <action>
# {
#     "name": "Search",
#     "arguments": {
#         "queries": "your search queries"
#     },
# }
# </action>
# Attention! Each search query needs to be simple and clear. For complex question or which requires reasoning, break it down and search step by step at multiple times. Try to search for only one or two queries at a time, unless there is strong need parallel searches. Each returned page is only a snippet. If you find any valuable pages, please use other browsing tool to get detailed information one by one!"""
#     inputs = {"queries": {"type": "array", "description": "List of web search queries"}}
#     output_type = "string"
#     simplified_description = "A web search tool like google, give it a list of search queries and return several related pages."

#     def __init__(self, search_engine="toutiao", pages_memory={}, search_params={}):
#         super().__init__()
#         self.search_engine = search_engine  # ["cnbing", "usbing", "toutiao", "mix"]]
#         self.pages_memeory: dict = pages_memory
#         self.page_index = 1
#         self.search_params = search_params

#     def call_webgpt_final_prompt(self, gen_query_list, search_engine, search_params):
#         if search_engine == "toutiao":
#             abparams = {
#                 "search": {
#                     "seed_plugin_summary_topk": 10,
#                     "seed_plugin_summary_max_total_len": 128000,
#                     "seed_plugin_summary_tokenizer_name": "bbpe155k-add_webgpt_fc",
#                 }
#             }
#         else:
#             abparams = {
#                 "search": {
#                     "seed_plugin_summary_max_total_len": 128000,
#                     "browsing_only_use_bing_search": True,
#                     "browsing_add_extra_bing_search": True,
#                     "seed_plugin_summary_bing_summary_min_extra_length": -1,
#                     "seed_plugin_summary_bing_only_full_text": False,
#                     "seed_plugin_summary_bing_topk": 10,
#                     "browsing_use_bing_tier": "s1",
#                 },
#             }
#             if search_engine == "usbing":
#                 abparams["search"]["plugin_bing_search_mkt"] = "en-US"
#                 abparams["search"]["plugin_bing_search_setlang"] = "en"

#             # if "browsing_use_bing_tier" in search_params:
#             #     abparams["search"]["browsing_use_bing_tier"] = search_params["browsing_use_bing_tier"]

#         thought = {
#             "input_query": gen_query_list[0],
#             "fine_queries": gen_query_list,
#             "input_qa_history": gen_query_list,
#             "abparams": json.dumps(abparams),
#             "text_ctrl": {"has_intent": True, "get_full_text": True},
#             "count": 10,
#             "model_arch": "seed_doubao_base",
#             "security_ctrl": {"white_website_level": 10},
#             "video_ctrl": {"intent": None},
#             "ruyi_ctrl": {"use_ruyi": True},
#         }

#         body = {
#             "PluginThoughtList": [{"PluginName": "SearchPlugin", "Thought": json.dumps(thought, ensure_ascii=False).encode("utf8").decode()}],
#             "SearchCommonInfo": {
#                 "appId": "497858",
#                 "abParams": json.dumps(abparams),
#             },
#             "BizId": "seed",
#             "TrafficGroup": "NLP_LLM",
#             "TrafficId": "nlp_llm_offline",
#             # "TrafficId": "web_server",
#             "ak": "bHdA0TkREkEQP2c5cUXGw5rzEa2bYwdf",
#         }

#         headers = {"Content-Type": "application/json;charset=UTF-8"}
#         # headers = {"Content-Type": "application/json;charset=UTF-8", "x-tt-env": "ppe_seed_browsing_tyx", "x-use-ppe": "1"}

#         resp = requests.post("https://ah3yte2a.fn.bytedance.net/api/v1/observe?", json=body, headers=headers).json()

#         obs_list = resp.get("data", {}).get("plugin_observation_list", [])

#         if len(obs_list) == 1:
#             obsJ = json.loads(obs_list[0].get("Observation", ""))
#             pages = []
#             for page in obsJ["doc_results"]:
#                 core_content = ""
#                 if "core_content" in page:
#                     core_content = page["core_content"]
#                 elif "summary" in page:
#                     core_content = page["summary"]
#                 skip = False
#                 for banned_sitename in search_params.get("banned_sitename", []):
#                     if banned_sitename in page["url"]:
#                         skip = True
#                         break
#                 if skip:
#                     continue
#                 if search_engine == "toutiao":
#                     snippet = page["summary"] if len(page["summary"]) < 1000 else page["summary"][:1000] + "..."
#                 else:
#                     snippet = page["single_search_result_summary"]
#                 pages.append(
#                     {
#                         "url": page["url"].replace("https://arxiv.org/abs", "https://arxiv.org/pdf"),
#                         "title": page["title"],
#                         "publish_time": page["publish_time"],
#                         "snippet": snippet,
#                         "core_content": core_content[:5000],
#                         "sitename": page["sitename"],
#                     }
#                 )
#             return pages
#         else:
#             print("call webgpt final prompt error, gen_query_list=", gen_query_list)
#             return []

#     def search_pages(self, query):
#         if self.search_engine == "mix":
#             pages_toutiao = self.call_webgpt_final_prompt([query], search_engine="toutiao", search_params=self.search_params)
#             pages_bing = self.call_webgpt_final_prompt([query], search_engine="usbing", search_params=self.search_params)
#             pages = []
#             for i in range(max(len(pages_toutiao), len(pages_bing))):
#                 if len(pages_bing) > i:
#                     pages.append(pages_bing[i])
#                 if len(pages_toutiao) > i:
#                     pages.append(pages_toutiao[i])
#         else:
#             pages = self.call_webgpt_final_prompt([query], search_engine=self.search_engine, search_params=self.search_params)
#         return pages

#     def forward(self, queries: list) -> str:
#         page_num = 10 if len(queries) < 2 else 5
#         snippets = ""
#         for query in queries:
#             for _ in range(3):
#                 try:
#                     pages = self.search_pages(query)
#                     break
#                 except:
#                     pages = None
#                     time.sleep(1)
#             if pages is None:
#                 snippets += "There's no result from search query: {}. Perhaps change a query.\n".format(query)
#             else:
#                 snippets += "Result from search query: {}\n".format(query)
#                 for page in pages[:page_num]:
#                     page_idx = self.page_index
#                     snippets += "<page{}>:\ntitle:{}\nsitename:{}\npublish_time:{}\nurl:{}\nsnippet:{}\n".format(
#                         page_idx, page["title"], page["sitename"], page["publish_time"], page["url"], page["snippet"]
#                     )
#                     page["index"] = page_idx
#                     # self.pages_memeory[str(page_idx)] = page
#                     self.page_index += 1
#         return snippets


# class SearchPluginToolV3(SearchPluginToolV2):
#     description = """A web search tool like google, give it a search query and return several related pages. The calling example is as follow:
# <action>
# {
#     "name": "Search",
#     "arguments": {
#         "query": "your search query"
#     },
# }
# </action>
# Attention! Search query needs to be simple and clear. For complex question or which requires reasoning, break it down and search step by step at multiple times. Each returned page is only a snippet. If you find any valuable pages, please use other browsing tool to get detailed information one by one!"""
#     inputs = {"query": {"type": "string", "description": "A web search query to perform."}}
#     output_type = "string"

#     def __init__(self, search_engine="mix", pages_memory={}, search_params={}):
#         super().__init__(search_engine, pages_memory, search_params)
    
#     def __call__(self, queries: list, page_idx: int, sanitize_inputs_outputs=True) -> str:
#         return self.forward(queries, page_idx)

#     def forward(self, queries: list, page_idx: int) -> str:
#         page_num = 10 if len(queries) < 2 else 7     # 一个query给10个，多个query的话，每个query给7个
#         snippets = ""
#         for query in queries:
#             for _ in range(3):
#                 try:
#                     pages = self.search_pages(query)
#                     break
#                 except:
#                     pages = None
#                     time.sleep(5)
#             if pages is None:
#                 snippets += "There's no result from search query: {}. Perhaps change a query.\n".format(query)
#             else:
#                 #snippets += "Result from search query: {}\n".format(query)
#                 for page in pages[:page_num]:
#                     page_idx += 1
#                     snippets += "<|superscript|>:{}：\n本文标题：{}\n本文内容：{}\n本文链接：{}\n发布时间：{}\n".format(
#                         page_idx, page["title"], page["snippet"], page["url"], page["publish_time"]
#                     )
#                     page["index"] = page_idx
#                     self.pages_memeory[str(page_idx)] = page
#         results = {
#             'content':snippets.strip(), 
#             'page_idx': page_idx
#         }
#         return results

# class SearchPluginToolV4: #usbing with en, doubao search with cn
#     name = "Search"
#     description = """工具"搜索"（Search）介绍：
# - 功能说明：这是一个联网搜索工具，输入仅1个搜索问题，返回多个搜索结果，每个搜索结果以“搜索结果+序号”开始，包括标题、网页主要内容、网页发布时间等信息。
# - 使用场景：当你需要获取更多信息时，特别是目前已知信息存在如下问题时（1）关键信息缺失/矛盾，（2）数据时效性不足（3），存在更权威信源可能性，（4）可丰富回答维度，应该调用本工具进行信息的获取。请注意，本搜索工具是面向通用场景，如果想要搜索垂直领域，且正好有对应领域的垂直搜索工具，可以优先调用垂直搜索工具获得更好的结果，在还需要补充信息的情况下再调用本搜索工具进行补充。
# 要求：
# 1. 一次使用"搜索"的搜索词只能有一个。
# 2. 对于搜索，灵活判断使用中文还是英文的搜索词，因为query词可以使用中文和英文，对于一些可能在外文网站上的信息，优先使用英文搜索，对于其余信息则可以优先使用中文搜索。整体确保搜索query词长度不要过长（英文query词不限长度），query词不和已搜索过的query重复。
#  调用格式示例：
# <action>
# {
#     "name": "Search",
#     "arguments": {
#         "queries": ["BMW's equity structure 2025"]
#     },
# }
# </action>
# """

#     def __init__(self, use_mcp=True, use_abstract=True):
#         super().__init__()
#         self.use_mcp = use_mcp
#         self.use_abstract = use_abstract

#     def __call__(self, queries, page_idx, doc_ids=[], gids=[], max_total_len=16000):
#         abparams = {
#             "search": {
#                 "enable_deep_research_url_handle": True,
#                 "seed_plugin_summary_english_joint_i18n": True,
#                 "seed_plugin_summary_topk": 7,
#                 "seed_plugin_summary_global_search_topk": 7,
#                 "seed_plugin_summary_query_to_n_doc_rerank": 7,
#                 "seed_plugin_summary_fc_add_date_time": False,
#                 "seed_plugin_summary_fc_repeat_query": False,
#                 "browsing_add_extra_video_search": False,
#                 "enable_query_importance": False,
#                 "global_search_use_full_content": False,
#                 "enable_global_search_doubao_finance_stream": True,
#                 "enable_prompt_embed_local": True,
#                 "enable_doubao_universal_intents": True,
#                 "text2sql_char_limit": 3000,
#                 "text2sql_table_compress": True,
#                 "text2sql_add_origin_sql_v2": True,
#                 "doubao_global_search": {
#                     "search": {
#                         "doubao_enable_pdf": 1,
#                         "text2sql_model_desc_name": "18108",
#                         "open_small_doubao_text2sql": 1,
#                         "enable_text_2_sql_recall": 1,
#                         "new_engine_trace": 1,
#                         "aladdin_recall_timeout_ms": 1500,
#                         "aladdin_trace_timeout": 1300
#                     }
#                 },
#                 "doubao_vertical": {
#                     "search": {
#                         "monad": {
#                             "req_index_name_list": [
#                                 "doubao_finance",
#                                 "doubao_finance_pdf"
#                                 ],
#                             "doubao_vertical": {
#                                 "req_index_name_list": [
#                                     "doubao_finance",
#                                     "doubao_finance_pdf"
#                                 ]
#                             }
#                         }
#                     }
#                 }
#             },
#         }

#         if self.use_abstract:
#             abparams["search"].update(
#                 {
#                     "browsing_cn_bing_query_to_global_search": True,
#                     "global_search_use_full_content": True,
#                     "enable_doubao_summary_extract": True,
#                     "seed_plugin_max_text_doc_num": 50,
#                     "seed_plugin_max_video_doc_num": 50,
#                     "seed_plugin_global_search_doubao_summary_extract_mode": 1,
#                     "seed_plugin_global_search_summary_extract_mode": 1,
#                     "seed_plugin_douyin_summary_extract_mode": 1,
#                     "seed_plugin_summary_extract_global_search_doubao_doc_max_str_len": 200,
#                     "seed_plugin_summary_extract_global_search_doc_max_str_len": 500,
#                     "seed_plugin_summary_extract_douyin_doc_max_str_len": 200,
#                     "seed_plugin_summary_extract_global_search_doubao_threshold": 1.2,
#                     "seed_plugin_summary_extract_global_search_threshold": 1.2,
#                     "seed_plugin_summary_extract_douyin_threshold": 1.2,
#                     "global_search": {
#                         "search": {
#                             "summary_model_downstream": True,
#                         }
#                     }
#                 }
#             )
#             abparams["search"]["doubao_global_search"]["search"]["summary_model_downstream"] = True

#         arguments = {
#             "input_query": queries[0],
#             "query": queries,
#             "model_config": {
#                 "final_prompt_version": "fc-v1",
#                 "max_total_len": max_total_len,
#                 "tokenizer_name": "bbpe155k-add_webgpt_fc"
#             },
#             "SearchCommonInfo": {
#                 "appId": "497858",
#                 "localeInfo": {
#                     "city": "北京",
#                     "district": "海淀"
#                 },
#                 "abParams": json.dumps(abparams),
#             },
#             "from_mcp_call": True,
#             "fc_search_ctrl": {"start_ref_idx": page_idx}
#         }
#         arguments_str = json.dumps(arguments)
#         data = {
#             "name": "Search",
#             "arguments": arguments_str,
#             "traffic_group": "NLP_LLM",
#             "traffic_id": "rlhf",
#         }

#         logid = None
#         content = ""
#         references = []
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
#                 content = result["content"]
#                 references = result["references"]
#                 break
#             except:
#                 time.sleep(2)

#         result = {
#             "content": content,
#             "gids": [],
#             "doc_ids": [],
#             "metric": {"logid": logid},
#             "page_idx": page_idx + len(references),
#             "content_tokens": len(tokenizer.encode(content))
#         }
#         return result


class GeneralSearchTool:
    name = "GeneralSearch"
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
    
    def get_response(self, is_cot, search_request_list: list):
        if is_cot:
            for req in search_request_list:
                req['pagination'] = {"limit":8}    # 一个query8个doc
        else:
            for req in search_request_list:
                req['pagination'] = {"limit":15}    # 一个query6个doc，每个doc300token
                req['snippet'] = {"max_length": 300}

        #input_params = json.dumps({"search_request_list":search_request_list})
        #data = {"name": "GeneralSearch", "input_params": input_params, "api_id": "6268"}
        data = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "GeneralSearch",
                "arguments": {
                    "search_request_list": search_request_list
                }
            }
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code != 200:
            return None, {"status_message": f"Failed to call API Hub: {response.status_code} {response.text} {response.headers}"}
        log_id, result = response.headers.get("X-Tt-Logid"), response.json()
        search_result = json.loads(result['result']['content'][0]['text'])['result']
        #search_result = json.loads(result.get("data", {}).get("result", {}))
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
    
    def adapt_to_text(self, search_result: dict, page_idx: int, refid_url_map: dict, url_tosurl_map: dict, tosurl_url_map: dict, no_display_url_map: dict, url_no_display_map: dict):
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
                
                refid_url_map["<|superscript|>:"+str(page_idx+1)] = url
                
                rsp_str += "<|superscript|>:{}：\n本文标题：{}\n本文内容：{}\n本文链接：{}\n发布时间：{}\n".format(
                    page_idx+1, 
                    doc_info.get("title", ""), 
                    ''.join([item.get("text", "") for item in doc_info.get("snippet", []) if item.get("type", "") == 'text']).strip(),  # snippet是图文list，如果是图片跳过，文本合并
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
        host = search_document.get("host_info", {}).get("host", "")
        if len(host) > 0:
            result.append(("host", host))

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
    
    def _single_search_result_prompt(self, function_name, search_time, search_request, search_response, page_idx, refid_url_map, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title=None, published_date=None):
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
        count = 0
        for i, doc in enumerate(search_response.get("documents", [])):
            if curr_title and curr_title[:15].lower().replace("’", "'") in doc.get("doc_info", {}).get("title", "").lower().replace("’", "'"):
                continue
            # 创建document元素
            publish_time_str = doc.get("doc_info", {}).get("publish_time", "")
            try:
                if published_date:
                    published_date_parsed = datetime.datetime.strptime(published_date, "%Y-%m-%d")
                    if len(publish_time_str) > 0:
                        publish_time = dateutil.parser.parse(publish_time_str)
                        # 如果publish_time在published_date之后，过滤
                        if publish_time > published_date_parsed:
                            continue
            except (ValueError, TypeError):
                continue
            
            count += 1
            if count == 6: break
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
                    if key == 'url':
                        url = value
            
            refid_url_map[reference_id] = url
        
        # 将XML转换为字符串并美化格式
        return self._transform_to_output(root), page_idx
    
    def adapt_to_xml(self, search_request_list: list, search_result: dict, page_idx: int, refid_url_map: dict, url_tosurl_map: dict, tosurl_url_map: dict, no_display_url_map: dict, url_no_display_map: dict, curr_title: str = None, published_date: str = None):
        results = list()
        search_time = datetime.datetime.now()
        for search_request, search_response in zip(search_request_list, search_result['search_response_list']):
            content, page_idx = self._single_search_result_prompt("GeneralSearch", search_time, search_request, search_response, page_idx, refid_url_map, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title, published_date)
            results.append(content)
        return "\n".join(results), page_idx

    def __call__(self, is_cot, search_request_list, page_idx=0, last_time=None, curr_title=None, refid_url_map={}, no_display_url_map={}, url_tosurl_map={}, tosurl_url_map={}, url_no_display_map={}):
        search_result, metrics = self.get_response(is_cot, search_request_list)

        if self.return_prompt_type == 'text':
            rsp_str, page_idx = self.adapt_to_text(search_result, page_idx, refid_url_map, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map)
        elif self.return_prompt_type == 'xml':
            rsp_str, page_idx = self.adapt_to_xml(search_request_list, search_result, page_idx, refid_url_map, url_tosurl_map, tosurl_url_map, no_display_url_map, url_no_display_map, curr_title, last_time)
            # search_time = datetime.datetime.now()
            # tool_prompt, _ = self.adapter.get_search_result_prompt_and_refids(
            #     "GeneralSearch", search_time, {"search_request_list": search_request_list}, search_result, curr_title
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
    general_search_tool = GeneralSearchTool(return_prompt_type="xml")
    search_request_list = [{'query': 'KV cache LLM serving characterization cloud provider arXiv'}]
    result = general_search_tool(is_cot=True, search_request_list=search_request_list, page_idx=0, last_time="2026-04-02", curr_title="TaxAgent: How Large Language Model Designs Fiscal Policy", no_display_url_map={}, url_tosurl_map={}, tosurl_url_map={}, url_no_display_map={}, refid_url_map={})
    print(result['content'])
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
if __name__ == "__main__":
    uni_test()