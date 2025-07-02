from typing import Dict, Any, List
from datetime import datetime, timedelta
from ..core.base_agent import BaseAgent
import numpy as np
from PIL import Image
import cv2
import dashscope
from http import HTTPStatus
import os
import json
import traceback
import requests
from pathlib import Path
import uuid

class EnvironmentGeneratorAgent(BaseAgent):
    def __init__(self, agent_id: str, controller):
        super().__init__(agent_id, controller)
        self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.dashscope_key:
            raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
        self.weather_states = ["sunny", "rainy", "cloudy", "foggy", "stormy"]
        self.current_weather = "sunny"
        self.time_of_day = datetime.now().strftime("%H:%M")
        
    def process_task(self):
        """处理环境生成任务"""
        if not self.current_task:
            raise ValueError("没有当前任务")
            
        task_type = self.current_task.get("type")
        
        if task_type == "scene_generation":
            return self.generate_scene()
        elif task_type == "weather_system":
            return self.generate_weather()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
            
    def generate_scene(self) -> Dict[str, Any]:
        """从文本描述生成游戏场景"""
        scene_prompt = self.current_task["scene_prompt"]
        
        # 使用Qwen-VL模型生成场景图
        try:
            dashscope.api_key = self.dashscope_key
            # 打印API调用信息
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在调用通义万相API生成图片...")
            print(f"提示词: {scene_prompt}")
            print(f"模型: wanx-v1 | 尺寸: 1024*1024")
            
            # 使用通义万相模型生成图像
            response = dashscope.ImageSynthesis.call(
                model='wanx-v1',
                prompt=f"游戏场景概念图：{scene_prompt}",
                n=1,
                size='1024*1024'  # 修正尺寸格式为1024*1024
            )
        except Exception as e:
            return {
                "scene_description": f"基于描述'{scene_prompt}'生成的默认场景",
                "scene_image": None,
                "status": "failed",
                "detail": f"API请求异常: {str(e)}"
            }
        
        # 解析通义万相API响应
        try:
            if not hasattr(response, 'output') or not hasattr(response.output, 'results'):
                raise ValueError("无效的API响应格式")
            
            if len(response.output.results) == 0:
                error_msg = getattr(response.output, 'message', '未知错误')
                raise ValueError(f"API未返回任何结果，错误信息: {error_msg}")
            
            image_url = response.output.results[0].url
            if not image_url:
                raise ValueError("API返回的图片URL为空")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 图片生成成功!")
            print(f"图片URL: {image_url}")
            
            # 下载图片到本地
            import requests
            from pathlib import Path
            import uuid
            
            image_name = f"{uuid.uuid4()}.png"
            local_path = Path("static/images") / image_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                response = requests.get(image_url)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
                    
                local_url = f"/static/images/{image_name}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 图片已保存到本地: {local_url}")
                
                return {
                    "scene_description": scene_prompt,
                    "scene_image": local_url,
                    "key_elements": self._analyze_scene_elements(scene_prompt),
                    "status": "completed",
                    "code": 200,
                    "message": "success"
                }
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 图片下载失败: {str(e)}")
                return {
                    "scene_description": scene_prompt,
                    "scene_image": image_url,  # 仍然返回原始URL作为fallback
                    "key_elements": self._analyze_scene_elements(scene_prompt),
                    "status": "completed",
                    "code": 200,
                    "message": "success",
                    "warning": f"图片下载失败，使用原始URL: {str(e)}"
                }
                
        except Exception as e:
            # 详细记录错误日志
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "api_response": str(response),
                "request_data": {
                    "model": "wanx-v1",
                    "prompt": f"游戏场景概念图：{scene_prompt}",
                    "size": "1024*1024"
                },
                "stack_trace": traceback.format_exc()
            }
            print("="*50 + " ERROR LOG " + "="*50)
            print(json.dumps(error_log, indent=2, ensure_ascii=False))
            print("="*100)
            
            return {
                "scene_description": response.output.choices[0].message.content[0]["text"] if 
                    hasattr(response, 'output') and 
                    hasattr(response.output, 'choices') and
                    len(response.output.choices) > 0 and
                    hasattr(response.output.choices[0].message, 'content') and
                    isinstance(response.output.choices[0].message.content, list) and
                    len(response.output.choices[0].message.content) > 0 and
                    isinstance(response.output.choices[0].message.content[0], dict) and
                    "text" in response.output.choices[0].message.content[0]
                    else f"基于描述'{scene_prompt}'生成的默认场景",
                "scene_image": "https://placehold.co/600x400?text=图片生成失败",
                "status": "failed",
                "detail": f"API响应解析错误: {str(e)}",
                "error_log": error_log,  # 将错误日志也返回给调用方
                "code": 500,
                "message": "image generation failed"
            }
        
        return {
            "scene_description": scene_prompt,
            "scene_image": image_url,
            "key_elements": self._analyze_scene_elements(scene_prompt),
            "status": "completed"
        }
        
    def _analyze_scene_elements(self, prompt: str) -> Dict[str, Any]:
        """从文本提示中分析关键场景元素"""
        elements = {
            "terrain": "多样化的地形",
            "objects": ["建筑", "植被", "装饰物"],
            "lighting": "动态光照",
            "special_effects": ["雾气", "粒子效果"],
            "time_of_day": self.time_of_day,
            "weather": self.current_weather
        }
        
        # 简单关键词分析
        if "森林" in prompt:
            elements["terrain"] = "森林地形"
            elements["objects"].extend(["树木", "灌木"])
        elif "城市" in prompt:
            elements["terrain"] = "城市地形" 
            elements["objects"].extend(["建筑", "道路"])
            
        return elements
        
    def generate_weather(self) -> Dict[str, Any]:
        """生成动态天气系统"""
        weather_type = self.current_task.get("weather_type", "random")
        if weather_type == "random":
            weather_type = np.random.choice(self.weather_states)
        
        # 更新当前天气和时间
        self.current_weather = weather_type
        self.time_of_day = datetime.now().strftime("%H:%M")
        
        # 获取天气效果
        effects = self._get_weather_effects(weather_type)
        
        # 添加时间影响
        hour = int(self.time_of_day.split(":")[0])
        if 6 <= hour < 18:  # 白天
            effects["light_intensity"] *= 1.2
        else:  # 夜晚
            effects["light_intensity"] *= 0.5
            
        return {
            "weather": weather_type,
            "time": self.time_of_day,
            "effects": effects,
            "status": "completed"
        }
        
    def _analyze_scene(self, edges):
        """分析场景特征并生成描述"""
        # 简化实现 - 实际项目中使用更复杂的CV算法
        return "生成基于输入图像的游戏场景"
        
    def _get_weather_effects(self, weather_type):
        """获取天气效果参数"""
        effects = {
            "sunny": {"light_intensity": 1.0, "particles": 0},
            "rainy": {"light_intensity": 0.7, "particles": 500},
            "cloudy": {"light_intensity": 0.8, "particles": 100},
            "foggy": {"light_intensity": 0.6, "particles": 300},
            "stormy": {"light_intensity": 0.5, "particles": 800}
        }
        return effects.get(weather_type, {})
