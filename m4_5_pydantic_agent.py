"""
[M4.5] PydanticAI 重写实验：天气查询 Agent

对比目标：
  同一个任务（天气查询 + 结构化输出），手写版 vs PydanticAI 版。
  聚焦三个核心差异：
    1. output_type  → 类型契约替代手写 json.loads + try/except
    2. deps 注入    → 依赖注入替代全局变量 / 闭包
    3. @agent.tool  → 装饰器注册替代 ToolRegistry 手写路由

安装依赖：
  pip install pydantic-ai httpx

运行方式：
  export ANTHROPIC_API_KEY="sk-ant-..."
  python m4_5_pydantic_agent.py
"""

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: 数据模型定义
# ══════════════════════════════════════════════════════════════════════════════

class WeatherCondition(str, Enum):
    """天气状况枚举 —— 用 Enum 而非裸字符串，让模型不能自由发明值"""
    SUNNY    = "sunny"
    CLOUDY   = "cloudy"
    RAINY    = "rainy"
    SNOWY    = "snowy"
    FOGGY    = "foggy"
    STORMY   = "stormy"


class WeatherReport(BaseModel):
    """
    output_type 契约：Agent 的返回值必须是这个结构。
    
    对比 M2.4 手写版：
      手写版返回 str，调用方自己 json.loads，自己校验字段存在性。
      PydanticAI 版：这个类就是契约，验证失败自动重试，永远不会给你 None 或缺字段。
    """
    city:          str            = Field(description="城市名称")
    temperature_c: float          = Field(description="摄氏温度", ge=-100, le=80)
    condition:     WeatherCondition = Field(description="天气状况")
    wind_speed_kmh: float         = Field(description="风速（公里/小时）", ge=0)
    humidity_pct:  int            = Field(description="湿度百分比", ge=0, le=100)
    summary:       str            = Field(description="一句话天气总结，适合展示给用户")

    # 🐍 Python 插播：@property 让计算属性像字段一样访问，不存储，每次计算
    @property
    def temperature_f(self) -> float:
        """华氏温度（按需计算，不占存储）"""
        return self.temperature_c * 9 / 5 + 32

    def __str__(self) -> str:
        return (
            f"📍 {self.city}\n"
            f"🌡  {self.temperature_c}°C ({self.temperature_f:.1f}°F)\n"
            f"🌤  {self.condition.value} | 💨 {self.wind_speed_kmh} km/h | 💧 {self.humidity_pct}%\n"
            f"📝 {self.summary}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: 依赖注入容器
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeatherDeps:
    """
    运行时依赖容器 —— 所有外部状态集中在这里，不用全局变量。
    
    对比 M2.4 手写版：
      手写版：工具函数通过闭包或全局变量访问 API key。
      PydanticAI 版：deps 在 agent.run() 时注入，工具函数通过 ctx.deps 访问。
      
    好处：测试时传 MockWeatherClient，生产时传 RealWeatherClient，
          工具函数代码一行不改。
    """
    http_client: httpx.AsyncClient   # 复用连接池，不在工具内部 new
    api_key:     str                 # 不同用户可以用不同 key
    base_url:    str = "https://api.open-meteo.com/v1"  # 可替换为 mock URL


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Agent 定义
# ══════════════════════════════════════════════════════════════════════════════

# 关键参数解读：
#   model        → 使用 Anthropic Claude（PydanticAI 支持多 provider，格式 "provider:model"）
#   deps_type    → 告诉类型系统 deps 的类型，ctx.deps 会有完整的类型提示
#   output_type  → 返回值契约，验证失败自动 retry（默认 retries=1）
weather_agent = Agent(
    "anthropic:claude-sonnet-4-5",
    deps_type=WeatherDeps,
    output_type=WeatherReport,
    instructions=(
        "You are a weather assistant. "
        "Use the get_coordinates tool first to convert city names to lat/lon, "
        "then use get_weather to fetch actual weather data. "
        "Always base your WeatherReport on real tool results, never fabricate data."
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: 工具注册
# 
# 对比 M2.4 手写版的工具注册：
#   手写版：registry.register("get_weather", fn, schema_dict)  ← 手写 JSON Schema
#   PydanticAI：@agent.tool 装饰器 + 函数签名 + docstring  ← 自动生成 Schema
#
# 关键：第一个参数 ctx: RunContext[WeatherDeps] 是注入点，不会变成工具 schema 的一部分。
#       其余参数（city_name, lat, lon 等）自动变成模型可传的参数。
# ══════════════════════════════════════════════════════════════════════════════

@weather_agent.tool
async def get_coordinates(ctx: RunContext[WeatherDeps], city_name: str) -> dict:
    """
    Convert a city name to geographic coordinates using a geocoding API.
    
    Args:
        city_name: Name of the city (e.g., "Beijing", "北京", "Tokyo")
    
    Returns a dict with latitude, longitude, and resolved city name.
    """
    # Open-Meteo 的免费 geocoding API，不需要 key
    # ctx.deps.http_client 是从外部注入的，可以在测试时换成 mock
    resp = await ctx.deps.http_client.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1, "language": "en", "format": "json"},
    )
    resp.raise_for_status()
    data = resp.json()

    if not data.get("results"):
        # 🐍 Python 插播：ModelRetry 是 PydanticAI 的特殊异常。
        #   raise ModelRetry(...) → 把错误信息注入 context，让模型重新思考后重试。
        #   对比手写版：你要自己构造 error dict 返回，然后在 loop 里处理。
        raise ModelRetry(f"City '{city_name}' not found. Try a different spelling or English name.")

    result = data["results"][0]
    return {
        "latitude":  result["latitude"],
        "longitude": result["longitude"],
        "city":      result.get("name", city_name),
        "country":   result.get("country", ""),
    }


@weather_agent.tool
async def get_weather(
    ctx: RunContext[WeatherDeps],
    latitude:  float,
    longitude: float,
) -> dict:
    """
    Fetch current weather data for given coordinates using Open-Meteo API.
    
    Args:
        latitude:  Geographic latitude (-90 to 90)
        longitude: Geographic longitude (-180 to 180)
    
    Returns current weather metrics including temperature, wind speed, and condition code.
    """
    resp = await ctx.deps.http_client.get(
        f"{ctx.deps.base_url}/forecast",  # base_url 从 deps 来，测试时可替换
        params={
            "latitude":             latitude,
            "longitude":            longitude,
            "current":              "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
            "wind_speed_unit":      "kmh",
            "timezone":             "auto",
        },
    )
    resp.raise_for_status()
    data    = resp.json()
    current = data["current"]

    # WMO weather code → 人类可读状态
    # 参考：https://open-meteo.com/en/docs#weathervariables
    code = current.get("weather_code", 0)
    condition_map = {
        range(0, 2):   "sunny",
        range(2, 4):   "cloudy",
        range(45, 68): "foggy",
        range(51, 68): "rainy",
        range(71, 78): "snowy",
        range(80, 83): "rainy",
        range(95, 100): "stormy",
    }
    condition = "cloudy"  # 默认值
    for code_range, label in condition_map.items():
        if code in code_range:
            condition = label
            break

    return {
        "temperature_c":   current["temperature_2m"],
        "wind_speed_kmh":  current["wind_speed_10m"],
        "humidity_pct":    current["relative_humidity_2m"],
        "condition":       condition,
        "weather_code":    code,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: 手写版（对比基准）
# 对比感受：完成同样的事情，手写版需要多少额外代码
# ══════════════════════════════════════════════════════════════════════════════

def handwritten_version_sketch():
    """
    这不是可运行的代码，只是展示手写版需要额外处理的部分。
    用于在 Review 环节和 PydanticAI 版做对比。
    """
    # 1. 工具注册（手动写 JSON Schema）
    tools = [{
        "name": "get_weather",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude":  {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
        }
    }]

    # 2. Tool-use Loop（手写）
    # while True:
    #     response = send_request(messages, tools=tools)
    #     if response["stop_reason"] == "end_turn": break
    #     for block in response["content"]:
    #         if block["type"] == "tool_use":
    #             result = execute_tool(block["name"], block["input"])
    #             messages = add_message(messages, "user", [tool_result])

    # 3. 解析输出（手写，可能 crash）
    # text = extract_text(response)
    # try:
    #     data = json.loads(text)           # ← 可能 JSONDecodeError
    # except json.JSONDecodeError:
    #     data = json.loads(strip_markdown(text))  # ← M2.2 踩过这个坑
    # city = data.get("city", "unknown")    # ← 字段可能不存在
    # temp = float(data.get("temperature_c", 0))  # ← 类型可能不对

    # 4. 全局依赖（测试时没法换掉）
    # API_KEY = os.environ["WEATHER_API_KEY"]  # ← 全局变量，测试噩梦
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PART 6: 入口
# ══════════════════════════════════════════════════════════════════════════════

async def query_weather(city: str) -> WeatherReport:
    """
    查询指定城市的天气。
    
    注意：deps 在这里构造并注入，agent 定义层完全不感知具体的 key 和 client。
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        deps = WeatherDeps(
            http_client=client,
            api_key=os.environ.get("WEATHER_API_KEY", "demo"),  # open-meteo 不需要 key
        )
        result = await weather_agent.run(
            f"What's the current weather in {city}?",
            deps=deps,
        )
        # result.output 的类型是 WeatherReport，不是 str
        # IDE 能给你完整的类型提示和补全
        return result.output


async def main():
    cities = ["Beijing", "Tokyo", "London"]

    print("=" * 60)
    print("M4.5 PydanticAI Weather Agent")
    print("=" * 60)

    for city in cities:
        print(f"\n⏳ 查询 {city} 天气...")
        try:
            report = await query_weather(city)

            # report 是 WeatherReport 实例，不是 dict，不是 str
            # 直接访问字段，有类型检查，IDE 有补全
            print(report)
            print(f"   [华氏温度换算已缓存: {report.temperature_f:.1f}°F]")

        except Exception as e:
            print(f"   ❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
