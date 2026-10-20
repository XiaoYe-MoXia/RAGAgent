import json
import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# 全局配置
load_dotenv()
DEFAULT_DATE = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
FINANCIAL_KEYWORDS = ["预算", "费用", "钱", "花费", "价格", "金额", "人均", "酒店费", "餐饮费", "门票费"]

# ==========================================
# 1. 天气查询工具（优化日志，避免重复）
# ==========================================
@tool
def get_weather_info(city: str, date: str = DEFAULT_DATE) -> str:
    """
    【行程规划专用】调用Open-Meteo接口查询指定城市的实时天气信息
    参数:
        city: 城市名称（如杭州、北京）
        date: 行程日期（格式YYYY-MM-DD，仅用于日志展示）
    返回:
        str: 结构化的实时天气信息字符串
    """
    tool_start = time.time()
    print(f"\n{'='*40} 开始调用天气查询工具 {'='*40}")
    print(f"📌 工具调用开始时间戳：{tool_start:.2f}")
    print(f"📌 工具调用参数：")
    print(f"   - city: {city}（用户指定）")
    print(f"   - date: {date}（行程日期）")
    print(f"{'='*90}")

    try:
        # 1. 地理编码获取经纬度
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "zh",
            "format": "json"
        }
        geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data = geo_resp.json()
        
        if not geo_data.get("results"):
            raise Exception(f"未找到城市：{city}")
        
        city_info = geo_data["results"][0]
        lat, lon = city_info["latitude"], city_info["longitude"]
        city_name_cn = city_info.get("name", city)
        
        print(f"✅ 定位成功：{city_name_cn} ({lat:.2f}, {lon:.2f})")
        
        # 2. 获取实时天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", 
                       "weather_code", "wind_speed_10m", "wind_direction_10m", "pressure_msl"],
            "timezone": "auto"
        }
        
        weather_resp = requests.get(weather_url, params=weather_params, timeout=10)
        weather_data = weather_resp.json()
        current = weather_data["current"]
        
        # 3. 天气代码转中文描述
        weather_codes = {
            0: "晴朗", 1: "主要晴朗", 2: "部分多云", 3: "阴天",
            45: "雾", 48: "雾凇",
            51: "毛毛雨", 53: "中度毛毛雨", 55: "强毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            95: "雷雨", 96: "雷雨伴冰雹", 99: "强雷雨伴冰雹"
        }
        weather_desc = weather_codes.get(current["weather_code"], "未知")
        
        # 4. 构造返回结果
        result_msg = (
            f"城市：{city_name_cn}\n"
            f"行程日期：{date}\n"
            f"实时温度：{current['temperature_2m']}℃\n"
            f"体感温度：{current['apparent_temperature']}℃\n"
            f"天气状况：{weather_desc}\n"
            f"相对湿度：{current['relative_humidity_2m']}%\n"
            f"风速：{current['wind_speed_10m']} km/h\n"
            f"气压：{current['pressure_msl']} hPa"
        )
        
        tool_end = time.time()
        tool_cost = tool_end - tool_start
        print(f"\n✅ 天气查询成功（耗时：{tool_cost:.2f}秒）：\n{result_msg}")
        return result_msg
    
    except Exception as e:
        tool_end = time.time()
        tool_cost = tool_end - tool_start
        result_msg = f"天气查询异常：{str(e)}"
        print(f"\n❌ 天气查询异常（耗时：{tool_cost:.2f}秒）：{result_msg}")
        return result_msg

# ==========================================
# 2. 预算计算工具（保留原有逻辑）
# ==========================================
@tool
def calculate_trip_budget(
    city: str,
    days: int,
    person_num: int,
    hotel_price_per_night: int,
    food_cost_per_day: int,
    attraction_fee: int
) -> str:
    """
    【行程规划专用】计算行程总预算
    """
    tool_start = time.time()
    print(f"\n{'='*40} 开始调用预算计算工具 {'='*40}")
    print(f"📌 工具调用开始时间戳：{tool_start:.2f}")
    print(f"📌 工具调用参数：")
    print(f"   - city: {city}")
    print(f"   - days: {days}")
    print(f"   - person_num: {person_num}")
    print(f"   - hotel_price_per_night: {hotel_price_per_night}")
    print(f"   - food_cost_per_day: {food_cost_per_day}")
    print(f"   - attraction_fee: {attraction_fee}")
    print(f"{'='*90}")

    if not all([city, days, person_num, hotel_price_per_night, food_cost_per_day, attraction_fee]):
        err_msg = "❌ 预算计算失败：缺少用户明确提供的参数，禁止自编参数"
        print(err_msg)
        return err_msg

    try:
        hotel_total = hotel_price_per_night * (days - 1) * person_num
        food_total = food_cost_per_day * days * person_num
        attraction_total = attraction_fee * person_num
        total = hotel_total + food_total + attraction_total
        
        result_msg = (
            f"【{city} {days}天行程预算明细】\n"
            f"1. 酒店费用：{hotel_total}元\n"
            f"2. 餐饮费用：{food_total}元\n"
            f"3. 景点门票：{attraction_total}元\n"
            f"总预算：{total}元"
        )
        tool_end = time.time()
        tool_cost = tool_end - tool_start
        print(f"\n✅ 预算计算成功（耗时：{tool_cost:.2f}秒）：\n{result_msg}")
        return result_msg
    except Exception as e:
        tool_end = time.time()
        tool_cost = tool_end - tool_start
        result_msg = f"预算计算异常：{str(e)}"
        print(f"\n❌ 预算计算异常（耗时：{tool_cost:.2f}秒）：{result_msg}")
        return result_msg

# ==========================================
# 3. 初始化模型（统一走 MiniMax OpenAI 兼容接口，与 weather.py 保持一致）
# ==========================================
llm = ChatOpenAI(
    model=os.getenv("MINIMAX_MODEL", "abab6.5s-chat"),
    temperature=0.3,
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
    timeout=60,  # 给工具调用 + 多步推理留足时间
)

# ==========================================
# 4. 创建智能体（核心优化：明确指令）
# ==========================================
agent = create_agent(
    model=llm,
    tools=[get_weather_info, calculate_trip_budget],
    system_prompt="""
    你是资深旅行规划师，严格遵守以下规则（违反直接作废）：
    1. 只提取用户明确提供的信息，绝不编造任何参数。
    2. 当用户说“从 A 去 B”时：
       - 必须调用 get_weather_info 查询【出发地 A】
       - 必须调用 get_weather_info 查询【目的地 B】
       - 两个城市都要查，各调用 1 次
    3. 调用 calculate_trip_budget 工具，输出出行每类花费的大致预算。
    4. 拿到两地天气结果后，立即输出完整的出行规划，不再调用工具、不重复思考。
    5. 规划必须包含：
       - 出发地天气
       - 目的地天气
       - 交通建议，具体到哪个交通工具，车站等等，越详细越好
       - 每日游玩路线，具体到哪个景点，怎么去，是否需要门票，怎么玩
       - 出行花费预算，分类别输出，如住宿、吃饭、门票
       - 天气适配的注意事项
    6. 输出格式清晰、实用、完整。
    7. 严格使用工具调用（tool calls）来查询天气和预算，绝不要在文本回复中写 `functions.xxx(...)` 这样的伪函数语法——必须通过工具调用通道完成任务。
    """
)

# ==========================================
# 5. 主程序（优化回复捕获+增加兜底）
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("✈️  智能行程规划助手（真实天气接口版）")
    print("=" * 60)
    
    # 用户输入
    user_input = input("请输入你的行程需求：\n").strip()
    
    if not user_input:
        print("❌ 错误：行程需求不能为空！")
    else:
        print(f"\n📝 用户输入的行程需求：\n{user_input}")
        print("\n" + "="*60)
        print("🤖 智能体开始处理，请稍候...")
        print("="*60)
        
        # 初始化变量
        print("\n📜 智能体思考与交互过程：")
        print("-"*60)
        final_output = ""
        chunk_count = 0
        tool_called = set()
        start_time = time.time()
        weather_data = ""  # 存储天气数据，用于兜底
        
        try:
            # 调用agent.stream
            print(f"\n🔄 开始调用agent.stream，当前时间：{time.time():.2f}")
            for chunk in agent.stream({
                "messages": [{"role": "user", "content": user_input}]
            }, stream_mode="values"):
                chunk_count += 1
                print(f"\n📦 处理第{chunk_count}个chunk")
                
                latest_msg = chunk["messages"][-1]
                
                # 1. 捕获工具调用和天气数据
                if isinstance(latest_msg, AIMessage):
                    # 提取工具调用记录
                    if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                        for tool_call in latest_msg.tool_calls:
                            tool_name = tool_call['name']
                            if tool_name not in tool_called:
                                tool_called.add(tool_name)
                                print(f"\n🔧 模型触发工具调用：")
                                print(f"   - 工具名：{tool_name}")
                                print(f"   - 调用参数：{json.dumps(tool_call['args'], ensure_ascii=False, indent=4)}")
                    
                    # 2. 提取工具返回结果（天气数据）
                    if hasattr(latest_msg, 'content') and "城市：" in latest_msg.content and "实时温度：" in latest_msg.content:
                        weather_data = latest_msg.content.strip()
                    
                    # 3. 提取最终规划（优先捕获完整规划）
                    if hasattr(latest_msg, 'content') and latest_msg.content.strip():
                        content = latest_msg.content.strip()
                        # 判定是否是最终规划（包含固定格式关键词）
                        if "###" in content and "【天气参考】" in content and "【每日游玩规划】" in content:
                            final_output = content
                        # 兜底：如果没有匹配格式，直接取最后一个非空内容
                        elif not final_output:
                            final_output = content

        except Exception as e:
            # 异常处理
            print(f"\n❌ 智能体处理异常：{str(e)}")
            print(f"❌ 异常类型：{type(e).__name__}")
        
        # 输出最终结果
        end_time = time.time()
        total_elapsed = end_time - start_time
        print("\n" + "="*60)
        print(f"📊 处理完成统计 - 总耗时：{total_elapsed:.2f}秒 | 工具调用数：{len(tool_called)} | Chunk数：{chunk_count}")
        print("📋 智能体最终回复：")
        print("="*60)
        print(final_output)