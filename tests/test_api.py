"""API 完整测试：5 个 endpoint 都跑一遍"""
import json
import sys
import time
import requests

BASE = "http://127.0.0.1:8000"


def show(name, resp):
    print(f"\n{'=' * 60}")
    print(f"【{name}】 {resp.request.method} {resp.url}")
    print(f"状态: {resp.status_code}")
    try:
        data = resp.json()
        text = json.dumps(data, ensure_ascii=False, indent=2)
        print(text[:2000])
    except Exception:
        print(resp.text[:500])


# 1. 健康检查
show("1. 健康检查", requests.get(f"{BASE}/api/health"))

# 2. Chain 模式
show("2. Chain 模式", requests.post(
    f"{BASE}/api/chat/chain",
    json={"query": "请问你们退货几天内必须寄回？"},
    timeout=60,
))

# 3. RAG 模式（关键测试：之前有 .md 不加载的 bug）
show("3. RAG 模式 - 咖啡机", requests.post(
    f"{BASE}/api/chat/rag",
    json={"query": "咖啡机的水泵是什么品牌？压力多少？"},
    timeout=60,
))
show("3b. RAG 模式 - 退货", requests.post(
    f"{BASE}/api/chat/rag",
    json={"query": "退货几天内必须寄回？"},
    timeout=60,
))

# 4. 多轮对话
show("4a. 多轮 1: 我叫张三", requests.post(
    f"{BASE}/api/chat/multi",
    json={"query": "我叫张三，今年 25 岁", "session_id": "test_api"},
    timeout=60,
))
show("4b. 多轮 2: 我叫什么？", requests.post(
    f"{BASE}/api/chat/multi",
    json={"query": "我叫什么？多大了？", "session_id": "test_api"},
    timeout=60,
))

# 5. 查历史
show("5. 查历史", requests.get(f"{BASE}/api/sessions/test_api/history"))

# 6. Agent
show("6. Agent 模式 - 问时间", requests.post(
    f"{BASE}/api/chat/agent",
    json={"query": "现在几点了？"},
    timeout=90,
))

# 7. 清空会话
show("7. 清空会话", requests.delete(f"{BASE}/api/sessions/test_api"))

print("\n" + "=" * 60)
print("测试完成！")
