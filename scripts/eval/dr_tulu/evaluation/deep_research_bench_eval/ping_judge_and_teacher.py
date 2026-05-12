"""快速探测两个外部依赖的可用性：

1. internal_http judge（RACE 评测用的 endpoint，
   默认 https://ivavmlgq.fn.bytedance.net，走 sys-proxy-rd-relay.byted.org:8118）
2. GPT-5.x Azure OpenAI（gpt_openapi），拿 --ak-list 里第一把 AK 验证

用法::

    uv run python scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/ping_judge_and_teacher.py \\
        --ak-list "$AK1,$AK2" \\
        --model-list "gpt-5-2025-08-07"

两个探测独立，出错会把原始异常类型和报文打出来，方便诊断
``Failed after 10 retries`` 这种被 swallowed 的报错。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import requests

DEFAULT_INTERNAL_JUDGE_URL = "https://ivavmlgq.fn.bytedance.net"
DEFAULT_INTERNAL_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
DEFAULT_AZURE_ENDPOINT = (
    "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
)
DEFAULT_AZURE_API_VERSION = "2024-03-01-preview"


def probe_internal_judge(url: str, proxy: str | None, timeout: int) -> None:
    print("\n=== Probe 1: internal_http judge ===")
    print(f"URL    : {url}")
    print(f"Proxy  : {proxy or '(none)'}")
    payload = {
        "messages": [
            {"role": "user", "content": "Ping. Reply with the single word: pong"}
        ]
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None
    t0 = time.time()
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            proxies=proxies,
            timeout=timeout,
        )
        dt = time.time() - t0
        print(f"HTTP   : {resp.status_code} ({dt:.2f}s)")
        body = resp.text
        print(f"Body   : {body[:400]}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"parsed.response: {(data.get('response') or '')[:200]!r}")
                print("RESULT : OK")
            except Exception as exc:  # noqa: BLE001
                print(f"RESULT : HTTP 200 but JSON parse failed: {exc}")
        else:
            print("RESULT : FAIL (non-200)")
    except requests.exceptions.ProxyError as exc:
        print(f"RESULT : FAIL (ProxyError) {exc}")
    except requests.exceptions.ConnectTimeout as exc:
        print(f"RESULT : FAIL (ConnectTimeout) {exc}")
    except requests.exceptions.ReadTimeout as exc:
        print(f"RESULT : FAIL (ReadTimeout after {timeout}s) {exc}")
    except requests.exceptions.SSLError as exc:
        print(f"RESULT : FAIL (SSLError) {exc}")
    except requests.exceptions.ConnectionError as exc:
        print(f"RESULT : FAIL (ConnectionError) {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT : FAIL ({type(exc).__name__}) {exc}")


def probe_azure_gpt5(
    ak: str, model: str, endpoint: str, api_version: str, timeout: int
) -> None:
    print("\n=== Probe 2: Azure GPT-5 (gpt_openapi) ===")
    print(f"Endpoint   : {endpoint}")
    print(f"api_version: {api_version}")
    print(f"Model      : {model}")
    print(f"AK tail    : ...{ak[-4:]}")
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        print("RESULT : FAIL (openai package not installed)")
        return

    try:
        client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=ak,
            timeout=float(timeout),
        )
        # GPT-5 的 reasoning tokens 也计入 max_completion_tokens，
        # 所以预算要大于 reasoning 预算 + 可见输出预算；这里给 2048 够用。
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_completion_tokens=2048,
            reasoning_effort="low",
            stream=False,
        )
        dt = time.time() - t0
        data = resp.model_dump()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        ctd = usage.get("completion_tokens_details") or {}
        reasoning_tokens = ctd.get("reasoning_tokens", 0) or 0
        visible_tokens = (usage.get("completion_tokens", 0) or 0) - reasoning_tokens
        print(f"Latency         : {dt:.2f}s")
        print(f"Finish reason   : {finish_reason}")
        print(f"Reasoning tokens: {reasoning_tokens}")
        print(f"Visible tokens  : {visible_tokens}")
        print(f"Content         : {content[:200]!r}")
        print(f"Usage           : {usage}")
        if content:
            print("RESULT : OK")
        elif finish_reason == "length" and visible_tokens == 0:
            print(
                "RESULT : API reachable but ran out of budget for visible content "
                "(all tokens went to reasoning). Increase --max-completion-tokens "
                "or lower --reasoning-effort. The key itself is OK."
            )
        else:
            print(f"RESULT : FAIL (empty content, finish_reason={finish_reason})")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT : FAIL ({type(exc).__name__}) {exc}")
        traceback.print_exc(limit=3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--judge-url",
        default=os.environ.get("DRB_INTERNAL_JUDGE_URL", DEFAULT_INTERNAL_JUDGE_URL),
    )
    p.add_argument(
        "--judge-proxy",
        default=os.environ.get("DRB_INTERNAL_PROXY", DEFAULT_INTERNAL_PROXY),
    )
    p.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable the judge proxy (tests direct connectivity).",
    )
    p.add_argument("--judge-timeout", type=int, default=60)

    p.add_argument(
        "--ak-list",
        default="",
        help="Comma-separated Azure API keys (will use the first one).",
    )
    p.add_argument("--model-list", default="gpt-5-2025-08-07")
    p.add_argument("--azure-endpoint", default=DEFAULT_AZURE_ENDPOINT)
    p.add_argument("--azure-api-version", default=DEFAULT_AZURE_API_VERSION)
    p.add_argument("--azure-timeout", type=int, default=60)
    p.add_argument(
        "--skip-judge", action="store_true", help="Skip the internal judge probe."
    )
    p.add_argument(
        "--skip-azure", action="store_true", help="Skip the Azure GPT-5 probe."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_judge:
        probe_internal_judge(
            url=args.judge_url,
            proxy=None if args.no_proxy else args.judge_proxy,
            timeout=args.judge_timeout,
        )

    if not args.skip_azure:
        aks = [x.strip() for x in args.ak_list.split(",") if x.strip()]
        models = [x.strip() for x in args.model_list.split(",") if x.strip()]
        if not aks:
            print("\n=== Probe 2: Azure GPT-5 ===\nSKIP: --ak-list is empty.")
        elif not models:
            print("\n=== Probe 2: Azure GPT-5 ===\nSKIP: --model-list is empty.")
        else:
            probe_azure_gpt5(
                ak=aks[0],
                model=models[0],
                endpoint=args.azure_endpoint,
                api_version=args.azure_api_version,
                timeout=args.azure_timeout,
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
