# weather.py 文件
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langchain_core.tools import tool

load_dotenv()

# Open-Meteo WMO 天气代码 -> 中文描述
WEATHER_CODE_CN = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "密集毛毛雨",
    56: "小冻雨",
    57: "密集冻雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "小冻雨",
    67: "大冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "猛烈阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _geocode(city: str) -> tuple[float, float, str]:
    """城市名 -> (纬度, 经度, 显示名)。Open-Meteo 支持中文/英文城市名。"""
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        raise ValueError(f"找不到城市：{city}")
    hit = results[0]
    return hit["latitude"], hit["longitude"], hit.get("name", city)


@tool
def get_weather(city: str) -> str:
    """Get the real-time weather for a given city (Open-Meteo, no API key required)."""
    try:
        lat, lon, display_name = _geocode(city)
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
                "timezone": "auto",
            },
            timeout=10,
        )
        r.raise_for_status()
        cw = r.json()["current_weather"]
        desc = WEATHER_CODE_CN.get(cw["weathercode"], f"天气代码 {cw['weathercode']}")
        return (
            f"{display_name} 当前天气：{desc}，气温 {cw['temperature']}°C，"
            f"风速 {cw['windspeed']} km/h。"
        )
    except Exception as e:
        return f"查询天气失败：{e}"


# 百度智能云千帆 / AIStudio 提供 OpenAI 兼容接口，用 ChatOpenAI 指向百度服务
model = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "ernie-5.0-thinking-preview"),
    base_url=os.getenv("LLM_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3"),
    api_key=os.getenv("LLM_API_KEY"),
)

agent = create_deep_agent(
    model=model,
    # tools 里面可以传入自定义函数，LangChain 工具或来自任何 MCP 服务器的工具
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)


def _extract_final_text(messages) -> str:
    """从 agent 返回的消息列表里倒着找第一条有文本内容的 AI 回复。"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") not in ("ai", "assistant"):
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""


if __name__ == "__main__":
    print("天气智能体已启动（输入 /quit 或按 Ctrl+C 退出）\n")
    messages: list = []
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit", "退出", "exit", "quit"}:
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        # 用框架返回的完整历史覆盖本地列表（包含 tool 调用等中间消息）
        messages = result["messages"]

        answer = _extract_final_text(messages)
        print(f"\nAI: {answer}\n")
