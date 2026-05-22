from __future__ import annotations

import ast
import asyncio
import concurrent
import difflib
import json
import math
import os
import random
import re
import traceback
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

import httpx
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from open_instruct import logger_utils

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    sent_tokenize = None

try:
    import ray
except ImportError:
    ray = None


logger = logger_utils.setup_logger(__name__)


organization_judge_instruction = """You are an intelligent, rigorous, and fair evaluator of scholarly writing quality and relevance.
You will receive the **title and abstract** of a research paper, together with
two candidate **related-work sections** (A and B) written for that paper.
Do not consider the formatting of the text e.g., latex , markdown, etc. Only consider the content.

Task: Decide which section—A or B—exhibits better organization and coherence.
Return only one letter: A or B.

How to judge (organization only)
Ignore breadth of coverage, citation accuracy, and analytic depth. Assess:

Logical structure – Clear introduction, grouping of related themes, and smooth progression of ideas.

Paragraph cohesion – Each paragraph develops a single topic and flows naturally to the next.

Clarity & readability – Minimal redundancy or contradictions; transitions guide the reader.

Signposting – Helpful headings, topic sentences, or discourse markers (if provided).

Pick the section that is easier to follow and better structured—no ties.

Output format: return a single token:
A – Section A is better organized.
B – Section B is better organized.

Output your answer as a **JSON dictionary** in the following format:
"decision": "A" or "B"
"explanation": "One sentence clearly explaining the key differences between the two options and why the selected one is preferred."
only output the dictionary, do not output any other text.

"""

citation_relevance_judge_instruction = """Given the title and abstract of a paper under assessment, the paper's ground-truth survey section (written by human experts), and the title and abstract of a candidate reference paper, determine the relevance of the candidate reference to the survey section.

Return a graded relevance score:
'2' – The reference is highly relevant to the survey section, reflecting prior work that directly addresses the problem or premise of the paper.
'1' – The reference is somewhat relevant to the survey section, reflecting prior work that is tangentially on-topic but does not directly address the problem or premises addressed in the paper.
'0' – The reference is irrelevant and should not be included in the survey section.

Instructions:
• Consider the main research topic and themes described in the survey section.
• Score 2 if the reference directly addresses similar problems, methods, or core concepts.
• Score 1 if the reference is in a related area but doesn't directly tackle the same problem (gray area - optional but reasonable to include).
• Score 0 if the reference is off-topic or unrelated in scope.

Remember: You are only seeing the title and abstract of the reference, so the full content might be more relevant than it appears.

Output your answer as a **JSON dictionary** in the following format:
"score": "2" or "1" or "0"
"explanation": "One sentence clearly explaining why the selected score is preferred."
only output the dictionary, do not output any other text.

"""

nugget_support_judge_instruction = """You are NuggetizeAssignerLLM, an intelligent assistant that can label a single atomic nugget based on if it is captured by a given response.

Return a single score:
'1' – The nugget is fully captured in the response.
'0' – The nugget is not captured in the response.

Output your answer as a **JSON dictionary** in the following format:
"score": "1" or "0"
"explanation": "One sentence clearly explaining why the selected score is preferred."
only output the dictionary, do not output any other text.

"""

citation_support_judge_instruction = """You are an intelligent and fair evaluator.
You are an Attribution Validator. Your task is to verify whether a given reference can support the given claim.

Return a single score:
'1' – The claim is fully supported by the reference.
'0' – The claim is not supported by the reference.

Output your answer as a **JSON dictionary** in the following format:
"score": "1" or "0"
"explanation": "One sentence clearly explaining why the selected score is preferred."
only output the dictionary, do not output any other text.

"""

question_judge_instruction = """You are an academic answer evaluator.

You will receive three inputs:

* **Academic Question**
* **Model Response**
* **Reference Answer**

Your task is to assign a **quality score between 0 and 1** to the Model Response, reflecting how well it answers the Academic Question compared to the Reference Answer.

---

## Evaluation Criteria

Evaluate the Model Response along the following dimensions:

1. **Relevance**
   Does the response directly address the academic question?

2. **Conceptual Correctness**
   Are the key concepts, definitions, claims, or mechanisms accurate?

3. **Coverage of Core Points**
   Does the response include the essential ideas contained in the Reference Answer?

4. **Logical Coherence**
   Is the reasoning clear, structured, and academically sound?

5. **Specificity and Substance**
   Does the response contain meaningful academic content rather than vague or generic statements?

---

## Scoring Guidelines (0–1 Continuous Scale)

* **1.0** → Fully correct, complete, and aligned with the reference answer.
* **0.8–0.9** → Mostly correct, minor omissions or slight imprecision.
* **0.6–0.7** → Partially correct, noticeable gaps or minor inaccuracies.
* **0.4–0.5** → Significant missing core content or conceptual errors.
* **0.2–0.3** → Largely incorrect or superficial.
* **0.0–0.1** → Irrelevant, incorrect, or does not answer the question.

The score should reflect **overall academic adequacy**, not stylistic elegance.

---

## Output Format

Return the result strictly in the following JSON format:

```json
{
  "score": <float between 0 and 1>,
  "justification": "<concise explanation of why this score was assigned>"
}
```

* The score must be a decimal number between 0 and 1.
* The justification must be brief (2–4 sentences).
* Do not output anything else.

"""

nugget_sp = """
你是一个严格的学术评测助手。你的任务是判断：**模型回复是否覆盖了指定的考点**。

---

### 输入包括：

* **用户问题（Question）**
* **指定考点（Key Point）**
* **模型回复（Answer）**

---

### 判定目标：

判断模型回复中是否**实质性地包含了指定考点的内容**，而不仅是表面相关或关键词重合。

---

### 判定标准：

#### 1. 语义覆盖（核心）

* 是否表达了考点的核心概念、原理或结论
* 允许同义改写，但必须语义等价

#### 2. 信息完整性

* 若考点包含多个关键要素，必须覆盖主要要素
* 缺失关键部分 → 判为未覆盖

#### 3. 显式优先

* 显式表达 → 覆盖
* 仅弱相关或间接推断 → 不覆盖

#### 4. 排除情况

以下一律视为未覆盖：

* 仅关键词匹配
* 泛泛相关但未触及核心
* 错误或矛盾内容

---

### 输出要求（极其重要）：

你**只能输出一个字符**，且必须严格符合以下规则：

* 只能输出：`0` 或 `1`
* **禁止输出任何其他内容**，包括但不限于：

  * 解释、理由
  * 标点符号（如句号、逗号）
  * 空格或换行之外的任何字符
  * JSON、文本、注释

---

### 合法输出示例：

```
1
```

```
0
```

---

### 非法输出示例（绝对禁止）：

```
1，因为回答包含了考点
```

```
0。
```

```
{"covered": 1}
```

```
答案是1
```

---

### 违规处理规则（自约束）：

* 如果你生成了除 `0` 或 `1` 之外的任何内容，则该输出视为**错误**
* 在输出前进行自检：确保最终输出**仅为单个字符**

---

### 判定原则：

* 宁严勿松：不确定时输出 `0`
* 以语义为准，不以关键词为准
* 判断“是否覆盖考点”，而不是“是否相关”
"""

rubrics_sp = """
## 1. 角色（Role）
你是一个正确率一直保持在100%的大模型评估专家，你擅长判断prompt下指定的一些要点，在模型回复中是否被满足

## 2. 核心任务（Core Task）
提供给你的信息包括：
- **prompt**：用户的问题
- **response**：某个大语言模型在这个prompt下的回复
- **要点列表**：列表中的每一条要点，都是由多名专家共同决策并撰写的，prompt下理想回答需要满足的条件

你需要完成的任务是：对于要点列表中的每一条要点，给出该要点在response中是否被满足的分析理由，以及满足/不满足的判断结论

## 3. 要点格式（Rubrics Format）
每一条要点由**要点编号**、**要点类型**、**要点内容**构成，每一个字段的含义如下：

### 3.1 要点编号
从1开始依次递增的整数

### 3.2 要点类型
取值为 必要点/重要点/附加点/误区 中的一种
#### 3.2.1 必要点
- 定义：response达到可用程度时必须满足的要点，体现出理想回答中最关键的一些要求
- 示例1：prompt=""京东快递客服电话""，要点列表中的一条必要点为
    ```json
    {""要点编号"": 1, ""要点类型"": ""必要点"", ""要点内容"": ""给到京东快递客服电话只为950616""}
    ```
- 示例2：prompt=""你认为虚拟主播经济和数字商业模式有什么联系""，要点列表中的一条必要点为
    ```json
    {""要点编号"": 1, ""要点类型"": ""必要点"", ""要点内容"": ""指出虚拟主播经济是指以2D、3D或AI驱动的虚拟形象为载体，进行内容创作、实时互动并实现商业变现的经济模式。""}
    ```

#### 3.2.2 重要点
- 定义：重要程度稍逊于必要点，缺失会降低用户满意度或结果完整性，但不会直接导致任务失败。
- 示例1：prompt=""环沪者富，环京者穷，原因是什么？建国初环京并不穷啊""，要点列表中的一条重要点为
    ```json
    {""要点编号"": 3, ""要点类型"": ""重要点"", ""要点内容"": ""明确北京的城市定位是全国政治中心、文化中心、国际交往中心、科技创新中心 。""}
    ```
- 示例2：prompt=""最近有新闻，一位女士踩到化学废弃物导致死亡，结合城市垃圾处理，回答如何解决""，要点列表中的一条必要点为
    ```json
    {""要点编号"": 5, ""要点类型"": ""重要点"", ""要点内容"": ""提及危险废弃物通常有独立的类别编码，严禁混入生活垃圾收运体系""}

#### 3.2.3 附加点
- 定义：重要程度属于锦上添花的一些内容，或是格式和风格的要求，只影响回答从“可用”到“完美”，不会导致回答从“不可用”变为“可用”
- 示例1：prompt=""对比一下越南和中国的区划""，要点列表中的一条附加点为
    ```json
    {""要点编号"": 12, ""要点类型"": ""附加点"", ""要点内容"": ""能够使用表格形式清晰呈现中国和越南行政区划调整的对比情况。表格应包括主要对比维度（如背景、原因、节奏、变化、影响、困难、争议），并体现两国在每个维度的特点及异同。表述应简明、条理清楚，便于直观理解和对比。""}
    ```

#### 3.2.4 误区/雷区
- 定义：从反面罗列出的一些错误说法、错误事实，满足误区类型的要点意味着response存在一些缺陷，不满足意味着response中未出现需要扣分的缺陷。
- 示例1：prompt=""四川GDP第二的城市是哪个""，要点列表中的一条误区为
    ```json
    {""要点编号"": 9, ""要点类型"": ""误区"", ""要点内容"": ""给四川省GDP第二的城市是绵阳市以外的说法""}
    ```
- 示例2：prompt=""华为GT6监测卵巢功能是智商税不""，要点列表中的一条误区为
    ```json
    {""要点编号"": 21, ""要点类型"": ""雷区"", ""要点内容"": ""给到的是针对于华为GT6以外的其他主体的分析""}
    ```

### 3.3 要点内容
要点的核心内容，即达成此要点的主要要求是什么

## 4. 要点满足判断（Rubrics Satisfy Judgement）
判断要点在response中是否满足时，需要遵守以下的原则：

### 4.1 原则1
在判断response是否满足要点前，需要充分理解**要点内容**的要求。
- 请严格遵守要点内容的要求执行，但**不**需要response和要点内容非常严格地匹配才算满足要点，没有明显相悖即可。

### 4.2 原则2
当要点内容较为宽泛，同时在不使用给出的要点信息以外的外部知识的情况下，难以判断要点是否满足时，请进行“善意假设”，即酌情放宽判断response满足要点的条件。以下通过几个例子说明：
- 例1：
    - prompt：第一次世界大战后的民族民主运动的特点
    - 要点：
        ```json
        {""要点编号"": 3, ""要点类型"": ""必要点"", ""要点内容"": ""给到>=1个一战后的民族民主运动的特点""}
        ```
    - 说明：如果回答中给到的特点是“追求民族独立与民主自由的目标明确”，尽管此时仅根据要点内容提供的信息无法确认这个特点是否符合要点要求，但是根据通用知识可以知道这个特点没有明显地和事实相悖，因此遵循“善意假设”原则，可以认为这个特点是符合要点要求的。
- 例2：
    - prompt：国内比较大的港口城市都有哪些
    - 要点：
        ```json
        {""要点编号"": 3, ""要点类型"": ""必要点"", ""要点内容"": ""给到>=3个中国国内规模较大的港口城市""}
        ```
    - 说明：如果回答中给到的城市是“济南”，应当判断为不满足要点要求。尽管要点内容提供的信息无法验证回答是否满足要点时应遵循“善意假设”，但是“善意假设”的前提是不能与基本的世界知识冲突。济南不是沿海城市，因此肯定不会是港口城市。

### 4.3 原则3
部分要点之间可能存在一定的关联，请保证这些相互关联的要点判断结果，不会存在明显的逻辑矛盾。

### 4.4 原则4
无论要点类型是什么，只需要根据要点内容判断是否满足。
- 示例1：
    - prompt：三氟甲磺酸锂，这个东西在固态电池当中有多重要？可替代性高吗？技术水平有多高？
    - 要点：
        ```json
        {""要点编号"": 2, ""要点类型"": ""雷区"", ""要点内容"": ""指出LiOTf适用于无机全固态电池（如硫化物、氧化物等陶瓷电解质体系）。""}
        ```
    - 说明：如果回答中明确指出LiOTf“并非氧化物、硫化物等无机固态电解质主流路线”，未提及三氟甲磺酸锂适用于无机全固态电池，则认为未满足雷区，因此该条要点的判断结果为""不满足""。
- 示例2：
    - prompt：暂停非美元计价的金属期权交易对黄金价格的影响，包括纸黄金和实物黄金
    - 要点：
        ```json
        {""要点编号"": 2, ""要点类型"": ""误区"", ""要点内容"": ""混淆了 LME（伦敦金属交易所）与 LBMA（伦敦金银市场协会）。""}
        ```
    - 说明：如果response区分了LME（针对有色金属期权）与伦敦场外市场/COMEX（黄金定价），未混淆LME与LBMA，未满足误区，因此该条要点的判断结果为""不满足""。
## 5. 输入格式
prompt:
```
{{用户的问题prompt}}
```
response:
```markdown
{{给出的大模型回复response}}
```
要点列表:
```json
{""要点编号"": 1, ""要点类型"": ""必要点"", ""要点内容"": ""xxx""}
{""要点编号"": 2, ""要点类型"": ""必要点"", ""要点内容"": ""xxx""}
...
{""要点编号"": N, ""要点类型"": ""必要点"", ""要点内容"": ""xxx""}
```

## 5. 输出格式
一行一个json，和要点列表中的要点**一一对应**，每个json里包括 要点编号, 判断理由, 判断结果 3个字段：
- 要点编号：需要和要点列表中的要点编号一一对应
- 判断结果：只能是“满足”或“不满足”，禁止使用其他表述
- 判断理由：判断结果对应的具体分析过程，注意要先给出判断理由再给出判断结果

具体格式如下:
```json
{""要点编号"": 1, ""分析理由"": ""xxx"", ""判断结果"": ""满足/不满足""}
{""要点编号"": 2, ""分析理由"": ""xxx"", ""判断结果"": ""满足/不满足""}
...
{""要点编号"": N, ""分析理由"": ""xxx"", ""判断结果"": ""满足/不满足""}
```
"""


def _extract_json_obj(text: str):
    """尽量鲁棒地从文本中提取 JSON 对象。"""
    if not isinstance(text, str):
        return None

    # 1) ```json {...}``` 或 ``` {...} ```
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2) 纯 JSON（整段就是）
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # 3) 文本中第一对 {...}
    left, right = text.find("{"), text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left : right + 1])
        except Exception:
            pass

    return None


def _extract_answer_score(text: str):
    """返回 [0,1] 内的分数；解析失败返回 None。"""
    obj = _extract_json_obj(text)
    if obj is None:
        return None
    try:
        val = float(obj.get("answer_score"))
    except Exception:
        return None
    return val


def clean_rich_media_reference(text: str, return_citation_ids: bool = False):
    """
    统一清洗候选答案里的引用标记，兼容两套协议：

    1. 新协议：<answer>...</answer> + <cite id="N">...</cite>
       会被转换成普通文本，其中引用统一替换为 [N]。

    2. 旧协议：<RichMediaReference>...</RichMediaReference> + superscript:X
       同样会被转换成普通文本，其中引用统一替换为 [X]。

    当 return_citation_ids=True 时，同时返回所有出现过的引用编号列表（去重，不保证按出现顺序）。
    为了结果稳定性，当前实现会按数字升序返回。
    """
    if not isinstance(text, str) or not text:
        if return_citation_ids:
            return "", []
        return ""

    # Sanitize: strip injected prompts and <answer> tags inside <think> blocks
    # so we only process the model's real answer block.
    text = _sanitize_candidate_for_format_check(text)

    citation_ids_set = set()

    def _maybe_add_citation_id(raw_id: str) -> int | None:
        s = str(raw_id)
        # Handle "call_<hex>-N" legacy format: extract trailing number after last '-'
        m = re.search(r"-(\d+)$", s)
        if m:
            cid = int(m.group(1)) + 1  # 0-indexed in call format -> 1-indexed doc id
            citation_ids_set.add(cid)
            return cid
        # Default: extract the last numeric sequence (handles "N", "<|superscript|>:N", etc.)
        match = re.search(r"(\d+)\s*$", s)
        if not match:
            match = re.search(r"(\d+)", s)
        if not match:
            return None
        try:
            cid = int(match.group(1))
        except Exception:
            return None
        citation_ids_set.add(cid)
        return cid

    # Prefer the new protocol when present: <answer>...</answer> with inline cites.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed <answer> at end of text (truncation by max_tokens).
    # If there are more opening <answer> tags than closing </answer> tags,
    # extract the trailing unclosed block as well.
    open_count = len(re.findall(r"<answer>", text, re.IGNORECASE))
    close_count = len(re.findall(r"</answer>", text, re.IGNORECASE))
    if open_count > close_count:
        # Find the last <answer> that has no matching </answer>
        last_open = text.rfind("<answer>")
        if last_open == -1:
            last_open = text.lower().rfind("<answer>")
        if last_open != -1:
            trailing_answer = text[last_open + len("<answer>") :]
            # Only use it if it has meaningful content (not just "..." placeholders)
            if len(trailing_answer.strip()) > 10:
                answer_matches.append(trailing_answer)

    if answer_matches:

        def _replace_cite(match):
            raw_id = match.group(1)
            inner = match.group(2) or ""
            cid = _maybe_add_citation_id(raw_id)
            suffix = f"[{cid}]" if cid is not None else ""
            return f"{inner}{suffix}"

        cleaned_answers = []
        for answer_text in answer_matches:
            cleaned_answers.append(
                re.sub(
                    r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>",
                    _replace_cite,
                    answer_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            )
        cleaned_text = "\n".join(part.strip() for part in cleaned_answers if part.strip())

        if return_citation_ids:
            return cleaned_text, sorted(citation_ids_set)
        return cleaned_text

    def _collect_ids(inner: str):
        # 匹配带逗号分隔的格式，如 <|superscript|>:12,<|superscript|>:15
        for x in re.findall(r"(?:<\|superscript\|>|superscript):(\d+)", inner.replace(",", " ")):
            _maybe_add_citation_id(x)

    def _replace(match):
        inner = match.group(1)
        _collect_ids(inner)
        # 替换所有 superscript:X 为 [X]，忽略逗号
        return re.sub(r"(?:<\|superscript\|>|superscript):(\d+)", r"[\1]", inner.replace(",", " "))

    cleaned_text = re.sub(r"<RichMediaReference>(.*?)</RichMediaReference>", _replace, text, flags=re.DOTALL)

    if return_citation_ids:
        return cleaned_text, sorted(citation_ids_set)
    return cleaned_text


def extract_document_infos_from_text(text: str) -> list[dict[str, Any]]:
    """从长文本中提取所有 <document>...</document> 的关键信息。

    输入示例（片段）：
    <document reference_id="<|superscript|>:1">
      <title>...</title>
      <summary>...</summary>
      ...
    </document>

    返回：
    [{"id": 1, "title": "...", "abstract": "..."}, ...]

    说明：abstract 优先取 <summary>；若不存在/为空则取 <snippet>。
    """

    if not isinstance(text, str) or not text:
        return []

    def _strip_or_empty(x: Any) -> str:
        return str(x).strip() if x is not None else ""

    def _element_text(elem: ET.Element | None) -> str:
        if elem is None:
            return ""
        return _strip_or_empty("".join(elem.itertext()))

    results: list[dict[str, Any]] = []

    # non-greedy: match each complete <document ...> ... </document>
    for m in re.finditer(r"<document\b.*?</document>", text, flags=re.DOTALL | re.IGNORECASE):
        doc_xml = m.group(0)

        # 1) extract id from reference_id="...:NUMBER"
        doc_id: int | None = None
        m_id = re.search(r"reference_id\s*=\s*\"[^\"]*?:(\d+)\"", doc_xml, flags=re.IGNORECASE)
        if m_id:
            try:
                doc_id = int(m_id.group(1))
            except Exception:
                doc_id = None

        title = ""
        abstract = ""

        # 2) parse XML to get <title> and <summary>/<snippet>
        try:
            root = ET.fromstring(doc_xml)
            title = _element_text(root.find("title"))
            abstract = _element_text(root.find("summary"))
            if not abstract:
                abstract = _element_text(root.find("snippet"))
        except Exception:
            # 3) fallback to regex if XML parsing fails
            m_title = re.search(r"<title>(.*?)</title>", doc_xml, flags=re.DOTALL | re.IGNORECASE)
            if m_title:
                title = _strip_or_empty(m_title.group(1))
            m_summary = re.search(r"<summary>(.*?)</summary>", doc_xml, flags=re.DOTALL | re.IGNORECASE)
            if m_summary:
                abstract = _strip_or_empty(m_summary.group(1))
            if not abstract:
                m_snippet = re.search(r"<snippet>(.*?)</snippet>", doc_xml, flags=re.DOTALL | re.IGNORECASE)
                if m_snippet:
                    abstract = _strip_or_empty(m_snippet.group(1))

        results.append({"id": doc_id, "title": title, "abstract": abstract})

    return results


SCORE_CORRECT = os.getenv("DEEP_RESEARCH_SCORE_CORRECT", "1")  # correct answers
SCORE_INCORRECT = os.getenv("DEEP_RESEARCH_SCORE_INCORRECT", "0")  # incorrect answers
SCORE_PENALTY = os.getenv("DEEP_RESEARCH_SCORE_PENALTY", "0")  # penalty for format errors
SCORE_ERROR = os.getenv("DEEP_RESEARCH_SCORE_ERROR", "-100000")  # llm_judge error

EXACT_MATCH = "Exact match."
ILLEGAL_CANDIDATE = "Illegal candidate."
JUDGE_ERROR = "Judge error."

PROXY_CONFIG = "http://sys-proxy-rd-relay.byted.org:8118"
FAAS_CONFIG_LST = [
    "https://78wyfcwp.fn.bytedance.net/arxiv_fetch",
    "https://n2vnfjcc.fn.bytedance.net/arxiv_fetch",
    "https://q9illr95.fn.bytedance.net/arxiv_fetch",
]

_CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*name\s*=\s*\"[^\"]+\"[^>]*)>(.*?)</call_tool>", re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CITE_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>", re.DOTALL | re.IGNORECASE)
# New format: <cite id="N">wrapped text</cite> (non-empty inner content)
_CITE_CONTENT_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.+?)</cite>", re.DOTALL | re.IGNORECASE)

# Regex to strip the FORCE_FINAL_ANSWER_PROMPT injected into the candidate text.
# This prompt contains literal <answer>...</answer> and <think>...</think> examples
# that confuse the format checks.  We match it loosely so minor edits don't break it.
_FORCE_PROMPT_RE = re.compile(
    r"You have reached the maximum number of allowed tool calls\..*?"
    r"instead of searching again\.",
    re.DOTALL,
)


def _sanitize_candidate_for_format_check(text: str) -> str:
    """Remove injected prompts so format checks see only model output.

    The ``FORCE_FINAL_ANSWER_PROMPT`` is appended to the response token
    stream by ``vllm_utils`` and contains literal ``<answer>...</answer>``
    / ``<think>...</think>`` examples that confuse the format checks.
    This function strips that injected text.
    """
    # Remove the FORCE_FINAL_ANSWER_PROMPT text.
    text = _FORCE_PROMPT_RE.sub("", text)
    return text


def _truncate_for_log(text: str, limit: int = 160) -> str:
    if not isinstance(text, str):
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _log_reward_debug(data_type: str, query: str, result_dict: dict[str, Any]) -> None:
    logger.info(
        (
            "scholar_reward_debug data_type=%s query=%r citation_len=%s "
            "reference_coverage_score=%s base_score=%s format_reward=%s "
            "format_weight=%s final_score=%s"
        ),
        data_type,
        _truncate_for_log(query),
        result_dict.get("citation_len"),
        result_dict.get("reference_coverage_score"),
        result_dict.get("base_score"),
        result_dict.get("format_reward"),
        result_dict.get("format_reward_weight"),
        result_dict.get("score"),
    )


def _real_format_reward_weight() -> float:
    raw = os.environ.get("SCHOLAR_REAL_FORMAT_REWARD_WEIGHT", "0.2")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.2
    return min(max(value, 0.0), 1.0)


def _protocol_format_reward(prediction: str) -> tuple[float, dict[str, bool]]:
    text = _sanitize_candidate_for_format_check(prediction or "")
    checks: dict[str, bool] = {}

    checks["has_call_tool"] = bool(_CALL_TOOL_RE.search(text))
    answer_matches = _ANSWER_RE.findall(text)
    checks["has_single_answer"] = len(answer_matches) == 1

    if answer_matches:
        # Use the last answer block for cite checking (most likely the real answer).
        answer_body = answer_matches[-1]
        checks["has_cite_inside_answer"] = bool(_CITE_CONTENT_RE.search(answer_body) or _CITE_RE.search(answer_body))
    else:
        checks["has_cite_inside_answer"] = False

    # Check tag balance. Exempt <think> mismatch only when the text was
    # clearly truncated (no </answer> ending, indicating token budget hit).
    call_balanced = text.count("<call_tool") == text.count("</call_tool>")
    answer_balanced = text.count("<answer>") == text.count("</answer>")
    think_balanced = text.count("<think>") == text.count("</think>")
    truncated = not text.rstrip().endswith("</answer>") and text.count("<think>") > text.count("</think>")
    checks["tags_balanced"] = call_balanced and answer_balanced and (think_balanced or truncated)

    # Check nesting: <answer> must NOT be inside <think>...</think>.
    # The last <answer> must appear AFTER the last </think>.
    last_think_close = text.rfind("</think>")
    last_answer_open = text.rfind("<answer>")
    checks["answer_not_nested"] = (
        last_answer_open > last_think_close if last_think_close >= 0 else True
    )

    score = sum(1 for v in checks.values() if v) / max(1, len(checks))
    return score, checks


def _apply_format_reward_blend(result_dict: dict[str, Any], candidate: str) -> None:
    base_score = sanitize_score(result_dict.get("score", 0.0))
    format_reward, format_checks = _protocol_format_reward(candidate)
    format_weight = _real_format_reward_weight()
    blended_score = sanitize_score((1.0 - format_weight) * base_score + format_weight * format_reward)

    result_dict["base_score"] = base_score
    result_dict["format_reward"] = format_reward
    result_dict["format_reward_weight"] = format_weight
    result_dict["format_reward_checks"] = format_checks
    result_dict["score"] = blended_score


def _invalid_reward_candidate_reason(candidate: str | None) -> str | None:
    if candidate is None:
        return None

    text = _sanitize_candidate_for_format_check(candidate).strip()
    if not text:
        return None

    answer_matches = _ANSWER_RE.findall(text)
    has_any_answer = len(answer_matches) >= 1
    has_tool_trace = (
        "<call_tool" in text
        or "<tool_output" in text
        or "</tool_output>" in text
        or "<tool_response" in text
        or "</tool_response>" in text
    )
    has_malformed_tool_trace = "<call_tool" in text and "</call_tool>" not in text

    # Only zero-out when the model truly produced NO answer after tool use.
    # Multiple answers are penalized via format_reward (partial credit) instead
    # of being treated as a fatal gate.
    if not has_any_answer and has_malformed_tool_trace:
        return "malformed_tool_trace_without_final_answer"
    if not has_any_answer and has_tool_trace:
        return "tool_trace_without_final_answer"
    return None


def _apply_invalid_candidate_zero_reward(result_dict: dict[str, Any], candidate: str | None, reason: str) -> None:
    _, format_checks = _protocol_format_reward(candidate or "")
    result_dict["invalid_candidate_reason"] = reason
    result_dict["base_score"] = 0.0
    result_dict["format_reward"] = 0.0
    result_dict["format_reward_weight"] = 0.0
    result_dict["format_reward_checks"] = format_checks
    result_dict["score"] = 0.0


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1), reraise=True)
async def _make_request_async(data) -> str:
    try:
        async with httpx.AsyncClient(proxy=PROXY_CONFIG) as client:
            response = await client.post(
                "https://ivavmlgq.fn.bytedance.net", json=data, headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise Exception(f"Request failed with status code {response.status_code}. Response: {response.text}")
            response_json = response.json()
            response_text = response_json.get("response", None)
            if not response_text:
                raise KeyError("'response' key not found in the response")
            return response_text

    except _make_request_async_cancel_errors() as e:
        print(f"[PT] [cancelerror] [soft_search_test_verifier] [make_request] ({e})")
        print(traceback.format_exc())
        raise AssertionError() from e


def _make_request_async_cancel_errors():
    errors = (asyncio.CancelledError, concurrent.futures.CancelledError)
    if ray is None:
        return errors
    return (ray.exceptions.TaskCancelledError, ray.exceptions.RayActorError, ray.exceptions.RayTaskError, *errors)


def get_arxiv_title_and_abstract(arxiv_id: str) -> tuple[str, str]:
    """Fetch title and abstract from arXiv by ID"""
    payload = {"arxiv_id": arxiv_id}
    faas_config = random.choice(FAAS_CONFIG_LST)
    response = requests.post(faas_config, json=payload)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}. Response: {response.text}")
        title, abstract = "Empty Title", "Empty Abstract"
    else:
        title, abstract = response.json()
    return title, abstract


def custom_sent_tokenize(text: str) -> list[str]:
    if sent_tokenize is None:
        return _split_by_en_punctuation(text)
    protected_text = text
    protected_text = re.sub(r"\bet al\.", "ET_AL_PLACEHOLDER", protected_text)
    sentences = sent_tokenize(protected_text)
    return [s.replace("ET_AL_PLACEHOLDER", "et al.") for s in sentences]


def _contains_cjk(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _split_by_cn_punctuation(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []

    end_punct = {"。", "！", "？", "；"}
    closing = {"”", "’", "」", "』", "）", ")", "]", "】", "》", "〉", '"', "'"}

    sentences: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch == "\n":
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
            i += 1
            continue

        if ch in end_punct:
            i += 1
            while i < n and text[i] in closing:
                buf.append(text[i])
                i += 1
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
            continue

        i += 1

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_by_en_punctuation(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    end_punct = {".", "!", "?", ";"}
    closing = {'"', "'", ")", "]", "}"}

    sentences: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch == "\n":
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
            i += 1
            continue

        if ch in end_punct:
            i += 1
            while i < n and text[i] in closing:
                buf.append(text[i])
                i += 1
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
            continue

        i += 1

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def custom_sent_tokenize_multilingual(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []

    cn_punct = {"。", "！", "？", "；"}
    has_cn_punct = any(p in text for p in cn_punct)
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_cjk = _contains_cjk(text)

    if not has_cn_punct and not has_cjk:
        if not has_latin:
            return [text.strip()] if text.strip() else []
        try:
            return [s.strip() for s in custom_sent_tokenize(text) if s.strip()]
        except Exception:
            return _split_by_en_punctuation(text)

    segments = _split_by_cn_punctuation(text)
    sentences: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if re.search(r"[A-Za-z]", seg):
            try:
                sentences.extend([s.strip() for s in custom_sent_tokenize(seg) if s.strip()])
            except Exception:
                sentences.extend(_split_by_en_punctuation(seg))
        else:
            sentences.append(seg)

    return sentences


def remove_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).replace(" |", "").strip()


def is_same_title(title1: str, title2: str) -> float:
    """
    判断两个标题是否相同，忽略大小写和空格。
    """
    if title1.strip().lower() == title2.strip().lower():
        return 1.0
    if title1.strip().lower() in title2.strip().lower() or title2.strip().lower() in title1.strip().lower():
        return 0.8
    return 0.0


def sanitize_score(x, default=0.0):
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return default
    return x


async def relevance_rate_evaluate(
    paper_title: str, section_title: str, gen_references: list[dict], text: str
) -> tuple[float, bool]:
    """
    评估生成的相关工作中引用的论文是否相关。

    :param paper_title: 论文标题。
    :param section_title: 相关工作标题。
    :param gen_references: 一个字典列表，每个字典代表一个引用的论文，包含标题和摘要。
    :param text: 真实的相关工作文本。
    :return: 一个元组，包含引用相关率分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """

    if not gen_references:
        return 0.0, True

    sp = citation_relevance_judge_instruction

    prompt_template = """### Paper under assessment\nSurvey Title: {paper_title}\nSection Title: {section_title}\n\n### Ground-truth survey section\n{text}\n\n### Candidate reference paper\nTitle: {ref_title}\nAbstract: {ref_abstract}\n"""

    score = 0.0
    is_valid = True
    response = None
    for ref in gen_references:
        ref_title = ref["title"]
        ref_abstract = ref["abstract"]

        try:
            prompt = prompt_template.format(
                paper_title=paper_title,
                section_title=section_title,
                text=text,
                ref_title=ref_title,
                ref_abstract=ref_abstract,
            )
            content = sp + prompt
            data = {"messages": [{"role": "user", "content": content}]}

            response = await _make_request_async(data)
            result = _extract_json_obj(str(response))
            score += float(result.get("score", 0.0)) / 2.0
        except Exception as e:
            print(f"RelevanceRateEvaluator Error processing ref_title: {ref_title}, response: {response}, error: {e}")
            is_valid = False

    score /= len(gen_references)
    score = sanitize_score(score)
    return score, is_valid


async def nugget_coverage_evaluate(nuggets: list[str], gen_response: str) -> tuple[float, bool]:
    """
    评估生成的相关工作是否覆盖了所有给定的nugget。

    :param nuggets: 一个字符串列表，每个字符串代表一个nugget。
    :param gen_response: 生成的相关工作文本。
    :return: 一个元组，包含覆盖率分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """

    if not nuggets:
        return 0.0, True

    sp = nugget_support_judge_instruction

    prompt_template = """### Nugget\n{nugget}\n\n### Response\n{gen_response}"""

    score = 0.0
    is_valid = True
    response = None

    for nugget in nuggets:
        prompt = prompt_template.format(nugget=nugget, gen_response=gen_response)
        content = sp + prompt
        data = {"messages": [{"role": "user", "content": content}]}

        try:
            response = await _make_request_async(data)
            result = _extract_json_obj(str(response))
            score += float(result.get("score", 0.0))
        except Exception as e:
            print(f"NuggetCoverageEvaluator Error processing nugget: {nugget}, response: {response}, error: {e}")
            is_valid = False
            continue

    score /= len(nuggets)
    score = sanitize_score(score)
    return score, is_valid


async def reference_coverage_evaluate(gen_references: list[dict], citation_list: list[str]) -> tuple[float, bool]:
    """
    评估生成的相关工作中引用的论文是否覆盖了所有给定的重要引用。

    :param gen_references: 一个字典列表，每个字典代表一个引用的论文，包含标题和摘要。
    :param citation_list: 一个字符串列表，每个字符串代表一个重要引用的标题（可能包含噪音）。
    :return: 一个元组，包含覆盖率分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """
    if citation_list == []:
        return 0.0, True

    def _normalize_title(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _best_substring_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_tokens = a.split()
        b_tokens = b.split()
        if not a_tokens or not b_tokens:
            return 0.0

        if len(a_tokens) > len(b_tokens):
            a, b = b, a
            a_tokens, b_tokens = b_tokens, a_tokens

        best = difflib.SequenceMatcher(None, a, b).ratio()
        a_len = len(a_tokens)
        min_len = max(1, a_len - 3)
        max_len = min(len(b_tokens), a_len + 3)

        for win_len in range(min_len, max_len + 1):
            for i in range(0, len(b_tokens) - win_len + 1):
                window = " ".join(b_tokens[i : i + win_len])
                sim = difflib.SequenceMatcher(None, a, window).ratio()
                if sim > best:
                    best = sim
        return best

    title_threshold = 0.88

    gen_titles_norm: list[str] = []
    for ref in gen_references:
        title = ref.get("title", "")
        gen_titles_norm.append(_normalize_title(title))

    score = 0.0
    for citation in citation_list:
        citation_norm = _normalize_title(citation)
        if not citation_norm:
            continue

        best = 0.0
        for gen_norm in gen_titles_norm:
            if not gen_norm:
                continue
            sim = _best_substring_similarity(citation_norm, gen_norm)
            if sim > best:
                best = sim
                if best >= title_threshold:
                    break

        if best >= title_threshold:
            score += 1.0

    score /= len(citation_list)
    score = sanitize_score(score)
    return score, True


async def citation_precision_evaluate(gen_response: str, gen_references: list[dict]) -> tuple[float, bool]:
    """
    评估生成的相关工作中引用的论文是否符合引用的精度。

    :param gen_response: 生成的相关工作文本。
    :param gen_references: 一个字典列表，每个字典代表一个引用的论文，包含标题和摘要。
    :return: 一个元组，包含引用精度分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """

    try:
        # Move trailing bracket citations *before* the sentence-ending
        # punctuation so they stay attached to the claim they support.
        # e.g. "Some claim.[1][6]" -> "Some claim[1][6]."
        normalised = re.sub(r"([.!?;])(\s*(?:\[\d+\])+)", r"\2\1", gen_response)
        sentences = custom_sent_tokenize_multilingual(normalised)
    except Exception as e:
        print(f"CitationPrecisionEvaluator Error while custom_sent_tokenize: {e}")
        return 0.0, False

    prompt_template = """### Reference Title\n{ref_title}\n\n### Reference Abstract\n{ref_abstract}"""
    references_prompt_dict = {
        int(ref["id"]): prompt_template.format(ref_title=ref["title"], ref_abstract=ref["abstract"])
        for ref in gen_references
    }

    sp = citation_support_judge_instruction

    citation_precision_scores = []
    is_valid = True
    response = None
    for sentence in sentences:
        if len(sentence) < 30:
            continue
        target = remove_citations(sentence)
        ref = [int(x[1:]) for x in re.findall(r"\[\d+", sentence)]
        for r in ref:
            if r in references_prompt_dict:
                prompt = f"""### Claim\n{target}\n\n""" + references_prompt_dict[r]
                content = sp + prompt
                data = {"messages": [{"role": "user", "content": content}]}

                try:
                    response = await _make_request_async(data)
                    result = _extract_json_obj(str(response))
                    score = float(result.get("score", 0.0))
                    citation_precision_scores.append(score)
                except Exception as e:
                    print(
                        f"CitationPrecisionEvaluator Error processing sentence: {sentence}, ref: {r}, response: {response}, error: {e}"
                    )
                    is_valid = False
                    continue
            else:
                # print(f"CitationPrecisionEvaluator: Reference {r} not found in gen_references.")
                citation_precision_scores.append(0.0)
                continue

    if not citation_precision_scores:
        print(
            f"CitationPrecisionEvaluator Error: No citation precision scores computed for gen_response:\n{gen_response}"
        )
        return 0.0, False

    score = np.mean(citation_precision_scores)
    score = sanitize_score(score)
    return score, is_valid


async def reference_appearance_evaluate(gen_response: str, gen_references: list[dict]) -> tuple[float, bool]:
    """
    评估生成的相关工作中引用的论文出现在相关工作中的比例。

    :param gen_response: 生成的相关工作文本。
    :param gen_references: 一个字典列表，每个字典代表一个引用的论文，包含标题和摘要。
    :return: 一个元组，包含引用出现率分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """

    # 提取gen_response中出现的所有[XX]格式引用
    citations = re.findall(r"\[\d+\]", gen_response)
    citations = [int(x[1:-1]) for x in citations]

    # 提取gen_references中的所有id
    gen_references_ids = [ref["id"] for ref in gen_references]

    # 计算引用出现率
    appearance_score_list = []
    for ref_id in gen_references_ids:
        if ref_id in citations:
            appearance_score_list.append(1.0)
        else:
            appearance_score_list.append(0.0)

    if not appearance_score_list:
        print(
            f"ReferenceAppearanceEvaluator Error: No reference appearance scores computed for gen_response:\n{gen_response}"
        )
        return 0.0, False

    score = np.mean(appearance_score_list)
    score = sanitize_score(score)
    return score, True


async def question_answer_evaluate(question: str, reference_answer: str, gen_response: str) -> tuple[float, bool]:
    """
    评估生成的回答是否与参考回答相符。

    :param question: 问题文本。
    :param reference_answer: 参考回答文本。
    :param gen_response: 生成的回答文本。
    :return: 一个元组，包含回答相符分数（0到1之间的浮点数）和一个布尔值，指示评估是否有效。
    """

    sp = question_judge_instruction

    prompt_template = """### Question\n{question}\n\n### Model Response\n{gen_response}\n\n### Reference Answer\n{reference_answer}"""

    score = 0.0
    is_valid = True
    response = None

    prompt = prompt_template.format(question=question, gen_response=gen_response, reference_answer=reference_answer)
    content = sp + prompt
    data = {"messages": [{"role": "user", "content": content}]}

    try:
        response = await _make_request_async(data)
        result = _extract_json_obj(str(response))
        score += float(result.get("score", 0.0))
    except Exception as e:
        print(f"QuestionAnswerEvaluator Error processing question: {question}, response: {response}, error: {e}")
        is_valid = False

    score = sanitize_score(score)
    return score, is_valid


async def evaluate_scholar_score_verifier(candidate: str, ground_truth: dict, tool_results_str: str) -> dict:
    # print(f'candidate: {candidate}')
    result_dict = {}
    result_dict["candidate"] = candidate
    result_dict["oracle_answer"] = "oracle_answer"
    result_dict["judge_response"] = "judge_response"
    result_dict["score"] = int(SCORE_PENALTY)

    gen_response, citation_ids = clean_rich_media_reference(candidate, True)
    document_infos = extract_document_infos_from_text(tool_results_str)
    gen_references = [doc for doc in document_infos if doc["id"] in citation_ids]
    result_dict["citation_len"] = len(gen_references)

    # --- Sampled trajectory debug print (5% of calls) ---
    _traj_sample_rate = float(os.environ.get("SCHOLAR_TRAJ_SAMPLE_RATE", "0.05"))
    if random.random() < _traj_sample_rate:
        _has_answer = bool(re.search(r"<answer>", candidate or "", re.IGNORECASE))
        _has_cite = bool(re.search(r"<cite\s", candidate or "", re.IGNORECASE))
        _invalid = _invalid_reward_candidate_reason(candidate) if candidate else "candidate_is_None"
        _doc_ids = sorted([d["id"] for d in document_infos])[:10]
        _cite_ids = sorted(citation_ids)[:10] if citation_ids else []
        # Extract a short sample of the answer block for inspection
        _ans_match = re.search(r"<answer>(.*?)</answer>", candidate or "", re.DOTALL | re.IGNORECASE)
        _ans_snippet = _ans_match.group(1)[:300] if _ans_match else "NO_ANSWER_BLOCK"
        # Show what reference_ids the model could see in the candidate text
        _ref_ids_in_candidate = re.findall(r'reference_id="([^"]{0,40})"', (candidate or "")[:5000])[:5]
        # Locate <cite> relative to <answer>
        _cite_positions = [m.start() for m in re.finditer(r"<cite\s", candidate or "", re.IGNORECASE)][:5]
        _answer_open_pos = [(m.start(), m.end()) for m in re.finditer(r"<answer>", candidate or "", re.IGNORECASE)]
        _answer_close_pos = [m.start() for m in re.finditer(r"</answer>", candidate or "", re.IGNORECASE)]
        # Show raw cite tags found in candidate
        _raw_cite_tags = re.findall(r"(<cite[^>]{0,80}>)", candidate or "", re.IGNORECASE)[:5]
        # Show candidate tail (last 800 chars) to see the actual answer area
        _candidate_tail = (candidate or "")[-800:]
        logger.info(
            "\n[TRAJ_DEBUG] === Sampled Trajectory ===\n"
            "  has_answer=%s, has_cite=%s, invalid_reason=%s\n"
            "  doc_ids_from_tool_results=%s\n"
            "  citation_ids_from_candidate=%s\n"
            "  gen_references_matched=%d\n"
            "  ref_ids_visible_to_model=%s\n"
            "  tool_results_str_len=%d, candidate_len=%d\n"
            "  answer_snippet=%r\n"
            "  answer_open_positions=%s, answer_close_positions=%s\n"
            "  cite_positions=%s\n"
            "  raw_cite_tags=%s\n"
            "  candidate_first_500=%r\n"
            "  candidate_tail_800=%r\n"
            "[TRAJ_DEBUG] === End ===",
            _has_answer,
            _has_cite,
            _invalid,
            _doc_ids,
            _cite_ids,
            len(gen_references),
            _ref_ids_in_candidate,
            len(tool_results_str or ""),
            len(candidate or ""),
            _ans_snippet,
            _answer_open_pos,
            _answer_close_pos,
            _cite_positions,
            _raw_cite_tags,
            (candidate or "")[:500],
            _candidate_tail,
        )

    if ground_truth["data_type"] == "understanding":
        query = ground_truth["verifiable_meta"]["query"]
        print(f"query: {query}")
        paper_title = ground_truth["verifiable_meta"]["paper_title"]
        answer = ground_truth["verifiable_meta"]["synthesis_answer"]
        citations_list = [paper_title]

        score_weight = {"nugget_coverage_score": 5.0, "reference_coverage_score": 3.0}

        for key in score_weight:
            result_dict[key] = 0.0

        if candidate is None:
            result_dict["score"] = int(SCORE_PENALTY)
        elif len(candidate.strip()) == 0:
            result_dict["score"] = int(SCORE_INCORRECT)
        elif invalid_reason := _invalid_reward_candidate_reason(candidate):
            _apply_invalid_candidate_zero_reward(result_dict, candidate, invalid_reason)
        else:
            score = int(SCORE_PENALTY)
            result_dict["reference_coverage_score"], _ = await reference_coverage_evaluate(
                gen_references, citations_list
            )
            result_dict["nugget_coverage_score"], _ = await question_answer_evaluate(query, answer, gen_response)

            ori_score = sum(result_dict[key] * score_weight[key] for key in score_weight)
            score = ori_score / sum(score_weight.values())
            score_str = f"score: {score:.4f}, " + ", ".join([f"{key}: {result_dict[key]:.4f}" for key in score_weight])
            print(score_str)

            if os.getenv("PT_DEBUG", None):
                print(
                    f"[PT] [scholar_score_verifier] judge {repr(gen_response[:100])} against ground truth {repr(answer[:100])}: {score_str}"
                )

            score = sanitize_score(score)
            result_dict["score"] = score
            _apply_format_reward_blend(result_dict, candidate)

        _log_reward_debug("understanding", query, result_dict)
        return result_dict, True
    elif ground_truth["data_type"] == "write_section":
        query = ground_truth["verifiable_meta"]["query"]
        print(f"query: {query}")
        paper_title = ground_truth["verifiable_meta"]["paper_title"]
        section_title = ground_truth["verifiable_meta"]["section_title"]
        text = ground_truth["verifiable_meta"]["text"]
        citations_list = ground_truth["verifiable_meta"]["citations"]
        nuggets = ground_truth["verifiable_meta"]["nuggets"]
        if isinstance(nuggets, str):
            nuggets = ast.literal_eval(nuggets)
        # citations_list = [citation['entry'] for citation in citations_list]

        score_weight = {
            "relevance_rate_score": 2.0,
            "nugget_coverage_score": 7.0,
            "reference_coverage_score": 10.0,
            "citation_precision_score": 1.0,
        }

        for key in score_weight:
            result_dict[key] = 0.0

        if candidate is None:
            result_dict["score"] = int(SCORE_PENALTY)
        elif len(candidate.strip()) == 0:
            result_dict["score"] = int(SCORE_INCORRECT)
        elif invalid_reason := _invalid_reward_candidate_reason(candidate):
            _apply_invalid_candidate_zero_reward(result_dict, candidate, invalid_reason)
        else:
            score = int(SCORE_PENALTY)
            result_dict["relevance_rate_score"], _ = await relevance_rate_evaluate(
                paper_title, section_title, gen_references, text
            )
            result_dict["reference_coverage_score"], _ = await reference_coverage_evaluate(
                gen_references, citations_list
            )
            result_dict["nugget_coverage_score"], _ = await nugget_coverage_evaluate(nuggets, gen_response)
            result_dict["citation_precision_score"], _ = await citation_precision_evaluate(
                gen_response, gen_references
            )

            ori_score = sum(result_dict[key] * score_weight[key] for key in score_weight)
            score = ori_score / sum(score_weight.values())
            score_str = f"score: {score:.4f}, " + ", ".join([f"{key}: {result_dict[key]:.4f}" for key in score_weight])
            print(score_str)

            if os.getenv("PT_DEBUG", None):
                print(
                    f"[PT] [scholar_score_verifier] judge {repr(gen_response[:100])} against ground truth {repr(text[:100])}: {score_str}"
                )

            score = sanitize_score(score)
            result_dict["score"] = score
            _apply_format_reward_blend(result_dict, candidate)

        _log_reward_debug("write_section", query, result_dict)
        return result_dict, True
    elif ground_truth["data_type"] == "write_survey":
        query = ground_truth["verifiable_meta"]["query"]
        print(f"query: {query}")
        paper_title = ground_truth["verifiable_meta"]["paper_title"]
        citations_list = ground_truth["verifiable_meta"]["citations"]
        sections_nuggets = ground_truth["verifiable_meta"]["nuggets"]
        if isinstance(sections_nuggets, str):
            sections_nuggets = ast.literal_eval(sections_nuggets)
        # print(f'sections_nuggets: {sections_nuggets}')
        nuggets = []
        for section_nugget_list in sections_nuggets:
            nuggets.append(section_nugget_list["title"])
            nuggets.extend(ast.literal_eval(section_nugget_list["nuggets"]))
        # print(type(nuggets))
        # print(f'nuggets: {nuggets}')
        if isinstance(nuggets, str):
            nuggets = ast.literal_eval(nuggets)
        # citations_list = [citation['entry'] for citation in citations_list]

        score_weight = {
            "nugget_coverage_score": 7.0,
            "reference_coverage_score": 10.0,
            "citation_precision_score": 1.0,
        }

        for key in score_weight:
            result_dict[key] = 0.0

        if candidate is None:
            result_dict["score"] = int(SCORE_PENALTY)
        elif len(candidate.strip()) == 0:
            result_dict["score"] = int(SCORE_INCORRECT)
        elif invalid_reason := _invalid_reward_candidate_reason(candidate):
            _apply_invalid_candidate_zero_reward(result_dict, candidate, invalid_reason)
        else:
            score = int(SCORE_PENALTY)
            result_dict["reference_coverage_score"], _ = await reference_coverage_evaluate(
                gen_references, citations_list
            )
            result_dict["nugget_coverage_score"], _ = await nugget_coverage_evaluate(nuggets, gen_response)
            result_dict["citation_precision_score"], _ = await citation_precision_evaluate(
                gen_response, gen_references
            )

            ori_score = sum(result_dict[key] * score_weight[key] for key in score_weight)
            score = ori_score / sum(score_weight.values())
            score_str = f"score: {score:.4f}, " + ", ".join([f"{key}: {result_dict[key]:.4f}" for key in score_weight])
            print(score_str)

            if os.getenv("PT_DEBUG", None):
                print(
                    f"[PT] [scholar_score_verifier] judge {repr(gen_response[:100])} against ground truth {repr(text[:100])}: {score_str}"
                )

            score = sanitize_score(score)
            result_dict["score"] = score
            _apply_format_reward_blend(result_dict, candidate)

        _log_reward_debug("write_survey", query, result_dict)
        return result_dict, True
    elif ground_truth["data_type"] == "search":
        query = ground_truth["verifiable_meta"]["query"]
        print(f"query: {query}")
        paper_title = ground_truth["verifiable_meta"]["paper_title"]
        section_title = ground_truth["verifiable_meta"]["section_title"]
        text = ground_truth["verifiable_meta"]["text"]
        citations_list = ground_truth["verifiable_meta"]["citations"]
        # citations_list = [citation['entry'] for citation in citations_list]

        score_weight = {"relevance_rate_score": 2.0, "reference_coverage_score": 10.0, "citation_precision_score": 1.0}

        for key in score_weight:
            result_dict[key] = 0.0

        if candidate is None:
            result_dict["score"] = int(SCORE_PENALTY)
        elif len(candidate.strip()) == 0:
            result_dict["score"] = int(SCORE_INCORRECT)
        elif invalid_reason := _invalid_reward_candidate_reason(candidate):
            _apply_invalid_candidate_zero_reward(result_dict, candidate, invalid_reason)
        else:
            score = int(SCORE_PENALTY)
            result_dict["relevance_rate_score"], _ = await relevance_rate_evaluate(
                paper_title, section_title, gen_references, text
            )
            result_dict["reference_coverage_score"], _ = await reference_coverage_evaluate(
                gen_references, citations_list
            )
            result_dict["citation_precision_score"], _ = await citation_precision_evaluate(
                gen_response, gen_references
            )

            ori_score = sum(result_dict[key] * score_weight[key] for key in score_weight)
            score = ori_score / sum(score_weight.values())
            score_str = f"score: {score:.4f}, " + ", ".join([f"{key}: {result_dict[key]:.4f}" for key in score_weight])
            print(score_str)

            if os.getenv("PT_DEBUG", None):
                print(
                    f"[PT] [scholar_score_verifier] judge {repr(gen_response[:100])} against ground truth {repr(text[:100])}: {score_str}"
                )

            score = sanitize_score(score)
            result_dict["score"] = score
            _apply_format_reward_blend(result_dict, candidate)

        _log_reward_debug("search", query, result_dict)
        return result_dict, True
    else:
        return {}, False


def extract_documents_from_text(text: str) -> list[dict[str, Any]]:
    """从长文本中提取所有 <document>...</document> 的信息。

    输入示例（片段）：
    <document reference_id="<|superscript|>:1">
      ...content...
    </document>

    返回：
    [{"id": 1, "content": "..."}, ...]

    """

    if not isinstance(text, str) or not text:
        return []

    def _strip_or_empty(x: Any) -> str:
        return str(x).strip() if x is not None else ""

    def _element_text(elem: ET.Element | None) -> str:
        if elem is None:
            return ""
        return _strip_or_empty("".join(elem.itertext()))

    results: list[dict[str, Any]] = []

    # non-greedy: match each complete <document ...> ... </document>
    for m in re.finditer(r"<document\b.*?</document>", text, flags=re.DOTALL | re.IGNORECASE):
        doc_xml = m.group(0)

        # 1) extract id from reference_id="...:NUMBER"
        doc_id: int | None = None
        m_id = re.search(r"reference_id\s*=\s*\"[^\"]*?:(\d+)\"", doc_xml, flags=re.IGNORECASE)
        if m_id:
            try:
                doc_id = int(m_id.group(1))
            except Exception:
                doc_id = None

        results.append({"id": doc_id, "content": doc_xml})

    return results


def _extract_json_objects_from_lines(text: str) -> list[dict]:
    results = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = _extract_json_obj(line)
        if obj is not None:
            results.append(obj)
    if results:
        return results
    pattern = re.compile(r'\{[^{}]*"要点编号"[^{}]*"判断结果"[^{}]*\}', re.DOTALL)
    for m in pattern.finditer(text):
        try:
            obj = json.loads(m.group())
            results.append(obj)
        except Exception:
            pass
    return results


def _majority_vote(values: list) -> Any:
    if not values:
        return None
    counter = Counter(values)
    return counter.most_common(1)[0][0]


_NEGATIVE_TYPES = {"误区", "雷区", "一级误区", "二级误区"}


def _classify_to_rubrics_type(classification: str) -> str:
    mapping = {
        "必要点": "必要点",
        "重要点": "重要点",
        "附加点": "附加点",
        "误区": "误区",
        "雷区": "雷区",
        "一级误区": "误区",
        "二级误区": "误区",
    }
    return mapping.get(classification, "必要点")


async def _eval_nuggets_objective(
    query: str, nuggets: list[dict], candidate_with_references: str
) -> dict[str, int | None]:
    prompt_template = """###用户问题\n{query}\n\n###指定考点\n{nugget}\n\n###模型回复\n{gen_response}"""
    results = {}
    for nugget in nuggets:
        nid = nugget["要点编号"]
        prompt = prompt_template.format(query=query, nugget=nugget["要点"], gen_response=candidate_with_references)
        content = nugget_sp + prompt
        data = {"messages": [{"role": "user", "content": content}]}
        try:
            response = await _make_request_async(data)
            results[nid] = int(response)
        except Exception as e:
            print(f"ScholarScoreEvalVerifier objective Error processing nugget: {nugget['要点编号']}, error: {e}")
            results[nid] = None
    return results


async def _eval_nuggets_subjective(
    query: str, nuggets: list[dict], candidate_with_references: str
) -> dict[str, int | None]:
    subj_nuggets = [n for n in nuggets if n.get("主客观") == "主观"]
    if not subj_nuggets:
        return {}
    idx_to_nid = {}
    nuggets_lines = []
    for seq_idx, nugget in enumerate(subj_nuggets):
        rubrics_idx = seq_idx + 1
        idx_to_nid[rubrics_idx] = nugget["要点编号"]
        rubrics_type = _classify_to_rubrics_type(nugget.get("重要性分类", "必要点"))
        nuggets_lines.append(
            json.dumps(
                {"要点编号": rubrics_idx, "要点类型": rubrics_type, "要点内容": nugget["要点"]}, ensure_ascii=False
            )
        )
    nuggets_text = "\n".join(nuggets_lines)
    prompt = f"prompt:\n```\n{query}\n```\nresponse:\n```markdown\n{candidate_with_references}\n```\n要点列表:\n```json\n{nuggets_text}\n```"
    content = rubrics_sp + prompt
    data = {"messages": [{"role": "user", "content": content}]}

    results = {}
    try:
        response = await _make_request_async(data)
        parsed_items = _extract_json_objects_from_lines(str(response))
        for item in parsed_items:
            rubrics_idx = item.get("要点编号")
            if rubrics_idx is None:
                continue
            if isinstance(rubrics_idx, str):
                try:
                    rubrics_idx = int(rubrics_idx)
                except ValueError:
                    continue
            nid = idx_to_nid.get(rubrics_idx)
            if nid is None:
                continue
            judgment = item.get("判断结果", "")
            is_satisfied = "满足" in judgment and "不满足" not in judgment
            nugget_cls = (
                subj_nuggets[rubrics_idx - 1].get("重要性分类", "必要点")
                if rubrics_idx <= len(subj_nuggets)
                else "必要点"
            )
            if nugget_cls in _NEGATIVE_TYPES:
                results[nid] = 0 if is_satisfied else 1
            else:
                results[nid] = 1 if is_satisfied else 0
    except Exception as e:
        print(f"ScholarScoreEvalVerifier subjective Error: {e}")

    for nugget in subj_nuggets:
        nid = nugget["要点编号"]
        if nid not in results:
            results[nid] = None
    return results


async def evaluate_scholar_score_eval_verifier(candidate: str, ground_truth: dict, tool_results_str: str) -> dict:
    result_dict = {}
    result_dict["candidate"] = candidate
    result_dict["oracle_answer"] = "oracle_answer"
    result_dict["judge_response"] = "judge_response"
    result_dict["score"] = int(SCORE_PENALTY)

    gen_response, citation_ids = clean_rich_media_reference(candidate, True)
    document_infos = extract_documents_from_text(tool_results_str)
    gen_references = [doc for doc in document_infos if doc["id"] in citation_ids]
    result_dict["citation_len"] = len(gen_references)
    candidate_with_references = gen_response
    for ref in gen_references:
        candidate_with_references += f"\n引用{ref['id']}: {ref['content']}"

    query = ground_truth["verifiable_meta"]["prompt"]
    nuggets = ground_truth["verifiable_meta"]["nuggets"]
    if isinstance(nuggets, str):
        nuggets = ast.literal_eval(nuggets)

    for idx, nugget in enumerate(nuggets):
        if "要点编号" not in nugget:
            nugget["要点编号"] = str(idx + 1)

    positive_weights = sum(float(n["考点权重"]) for n in nuggets if float(n["考点权重"]) > 0.0)
    if positive_weights == 0.0:
        result_dict["score"] = 0.0
        return result_dict, True

    if invalid_reason := _invalid_reward_candidate_reason(candidate):
        _apply_invalid_candidate_zero_reward(result_dict, candidate, invalid_reason)
        return result_dict, True

    obj_nuggets = [n for n in nuggets if n.get("主客观") == "客观"]
    subj_nuggets = [n for n in nuggets if n.get("主客观") == "主观"]
    obj_positive_weights = sum(float(n["考点权重"]) for n in obj_nuggets if float(n["考点权重"]) > 0.0)

    num_eval_rounds = 5

    obj_rounds: list[dict[str, int | None]] = []
    subj_rounds: list[dict[str, int | None]] = []
    eval_tasks = []
    for _ in range(num_eval_rounds):
        if obj_nuggets:
            eval_tasks.append(_eval_nuggets_objective(query, obj_nuggets, candidate_with_references))
        if subj_nuggets:
            eval_tasks.append(_eval_nuggets_subjective(query, nuggets, candidate_with_references))
    all_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

    has_obj = bool(obj_nuggets)
    has_subj = bool(subj_nuggets)
    tasks_per_round = int(has_obj) + int(has_subj)
    for i in range(num_eval_rounds):
        base = i * tasks_per_round
        task_idx = 0
        if has_obj:
            obj_res = all_results[base + task_idx]
            obj_rounds.append(obj_res if isinstance(obj_res, dict) else {n["要点编号"]: None for n in obj_nuggets})
            task_idx += 1
        if has_subj:
            subj_res = all_results[base + task_idx]
            subj_rounds.append(subj_res if isinstance(subj_res, dict) else {n["要点编号"]: None for n in subj_nuggets})

    def _is_valid_round(round_results: dict[str, int | None]) -> bool:
        return any(v is not None for v in round_results.values())

    def _compute_round_score(
        round_results: dict[str, int | None], all_rounds: list[dict[str, int | None]], target_nuggets: list[dict]
    ) -> float | None:
        if not _is_valid_round(round_results):
            return None
        score = 0.0
        for nugget in target_nuggets:
            nid = nugget["要点编号"]
            weight = float(nugget["考点权重"])
            k = round_results.get(nid)
            if k is None:
                other_values = [r.get(nid) for r in all_rounds if r.get(nid) is not None]
                k = _majority_vote(other_values) if other_values else 0
            score += weight * k
        return score

    def _get_nugget_majority_k(nid: str, all_rounds: list[dict[str, int | None]]) -> int:
        values = [r.get(nid) for r in all_rounds if r.get(nid) is not None]
        return _majority_vote(values) if values else 0

    score_all_weighted = 0.0
    for nugget in nuggets:
        nid = nugget["要点编号"]
        weight = float(nugget["考点权重"])
        if nugget.get("主客观") == "客观" and obj_rounds:
            k = _get_nugget_majority_k(nid, obj_rounds)
        elif nugget.get("主客观") == "主观" and subj_rounds:
            k = _get_nugget_majority_k(nid, subj_rounds)
        else:
            k = 0
        score_all_weighted += weight * k
    score_all = max(score_all_weighted, 0.0) / positive_weights

    if obj_nuggets and obj_positive_weights > 0.0:
        obj_raw = [_compute_round_score(r, obj_rounds, obj_nuggets) for r in obj_rounds]
        obj_scores = [s for s in obj_raw if s is not None]
        obj_final = _majority_vote([round(s, 6) for s in obj_scores]) if obj_scores else 0.0
        score_obj = max(obj_final, 0.0) / obj_positive_weights
    else:
        obj_scores = []
        obj_final = 0.0
        score_obj = 0.0

    if subj_nuggets:
        subj_raw = [_compute_round_score(r, subj_rounds, subj_nuggets) for r in subj_rounds]
        subj_scores = [s for s in subj_raw if s is not None]
        subj_final = (
            subj_scores[0]
            if len(subj_scores) == 1
            else (_majority_vote([round(s, 6) for s in subj_scores]) if subj_scores else 0.0)
        )
    else:
        subj_scores = []
        subj_final = 0.0

    result_dict["score"] = sanitize_score(score_all)
    result_dict["score_all"] = sanitize_score(score_all)
    result_dict["score_obj"] = sanitize_score(score_obj)
    result_dict["objective_scores"] = obj_scores
    result_dict["subjective_scores"] = subj_scores
    result_dict["objective_final"] = obj_final
    result_dict["subjective_final"] = subj_final

    return result_dict, True


def unit_test():
    gen_references = [
        {"title": "Structural Scaffolds for Citation Intent Classification in Scientific Publications"},
        {
            "title": "scite: A smart citation index that displays the context of citations and classifies their intent using deep learning"
        },
    ]

    important_citation_list = [
        "Structural Scaffolds for Citation Intent Classification in Scientific Publications",
        "Paper Title: Scite - a smart citation index that displays the context of citations and classifies their intent using deep learning (2021 preprint)",
    ]

    score, is_valid = asyncio.run(reference_coverage_evaluate(gen_references, important_citation_list))
    print("reference_coverage_evaluate score:", score, "is_valid:", is_valid)
    assert is_valid is True
    assert abs(score - 1.0) < 1e-12

    score2, is_valid2 = asyncio.run(reference_coverage_evaluate(gen_references, ["Totally Different Title"]))
    print("reference_coverage_evaluate score2:", score2, "is_valid:", is_valid2)
    assert is_valid2 is True
    assert abs(score2 - 0.0) < 1e-12
    return


if __name__ == "__main__":
    unit_test()
