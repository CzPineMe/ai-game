from typing import Dict, Any
from ..core.base_agent import BaseAgent
import os
import requests
import json
import time
import dashscope
from http import HTTPStatus

class ContentGeneratorAgent(BaseAgent):
    def __init__(self, agent_id: str, controller):
        super().__init__(agent_id, controller)
        self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.dashscope_key:
            raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
        if not self.deepseek_key:
            raise ValueError("未设置DEEPSEEK_API_KEY环境变量")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {self.deepseek_key}"
        }
        
    def process_task(self):
        """处理游戏内容生成任务"""
        if not self.current_task:
            raise ValueError("没有当前任务")
            
        task_type = self.current_task.get("type")
        
        if task_type == "storyline":
            return self.generate_storyline()
        elif task_type == "characters":
            return self.generate_characters()
        elif task_type == "elements":
            return self.generate_elements()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
            
    def generate_characters(self) -> Dict[str, Any]:
        """生成游戏角色设定"""
        try:
            # 输入验证
            if not self.current_task.get("prompt"):
                raise ValueError("缺少prompt参数")
                
            prompt = str(self.current_task["prompt"]).strip()[:500]
            character_type = self.current_task.get("character_type", "custom")
            count = min(max(int(self.current_task.get("count", 3)), 1), 10)  # 限制1-10个角色
            
            # 角色类型描述
            type_descriptions = {
                "hero": "英雄角色，包含正义感、成长历程",
                "villain": "反派角色，包含动机、阴谋",
                "support": "辅助角色，提供帮助或信息",
                "custom": "自定义类型角色"
            }
            
            if character_type not in type_descriptions:
                raise ValueError(f"不支持的角色类型: {character_type}")
            
            messages = [{
                "role": "system",
                "content": f"""你是一个专业的游戏角色生成器，擅长创作{type_descriptions.get(character_type)}风格的角色。
                请根据用户需求生成{count}个游戏角色，每个角色包含：
                - 姓名
                - 外貌描述
                - 背景故事
                - 性格特点
                - 动机和目标
                - 与其他角色的关系
                - 成长潜力
                - 游戏中的功能定位"""
            }, {
                "role": "user",
                "content": f"""游戏角色生成需求：
主题：{prompt}
角色类型：{character_type}
生成数量：{count}个

请生成详细角色设定，每个角色至少包含200字描述"""
            }]
            
            try:
                print("调用DeepSeek API生成角色...")
                data = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.85,
                    "max_tokens": 800 * count,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data
                )
                
                if response.status_code != HTTPStatus.OK:
                    return {
                        "error": f"API调用失败: {response.text}",
                        "status": "failed"
                    }
                    
                result = response.json()
                if "choices" not in result:
                    return {
                        "error": "API返回格式异常",
                        "status": "failed"
                    }
                    
                content = str(result["choices"][0]["message"]["content"])
                return {
                    "characters": content,
                    "count": count,
                    "status": "completed"
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "status": "failed"
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

    def generate_elements(self) -> Dict[str, Any]:
        """生成游戏元素(道具/技能/任务)"""
        try:
            if not self.current_task.get("prompt"):
                raise ValueError("缺少prompt参数")
                
            prompt = str(self.current_task["prompt"]).strip()[:500]
            element_type = self.current_task.get("element_type", "item")
            count = min(max(int(self.current_task.get("count", 3)), 1), 10)  # 限制1-10个元素
            
            type_descriptions = {
                "item": "游戏道具，包含名称、描述、使用效果、稀有度",
                "skill": "角色技能，包含名称、描述、效果、冷却时间、消耗",
                "quest": "游戏任务，包含任务名称、描述、目标、奖励"
            }
            
            if element_type not in type_descriptions:
                raise ValueError(f"不支持的元素类型: {element_type}")
            
            messages = [{
                "role": "system",
                "content": f"""你是一个专业的游戏元素生成器，擅长设计{type_descriptions.get(element_type)}。
                请根据需求生成{count}个游戏{type_descriptions.get(element_type)}，每个包含：
                - 名称
                - 详细描述
                - 游戏中的功能效果
                - 平衡性参数
                - 与其他元素的互动关系"""
            }, {
                "role": "user", 
                "content": f"""游戏元素生成需求：
主题：{prompt}
元素类型：{element_type}
生成数量：{count}个

请生成详细的游戏元素设定"""
            }]
            
            try:
                print(f"调用DeepSeek API生成{element_type}...")
                data = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.8,
                    "max_tokens": 600 * count,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data
                )
                
                if response.status_code != HTTPStatus.OK:
                    return {"error": f"API调用失败: {response.text}", "status": "failed"}
                    
                result = response.json()
                if "choices" not in result:
                    return {"error": "API返回格式异常", "status": "failed"}
                    
                content = str(result["choices"][0]["message"]["content"])
                return {
                    "elements": content,
                    "count": count,
                    "status": "completed"
                }
                
            except Exception as e:
                return {"error": str(e), "status": "failed"}
                
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def generate_storyline(self) -> Dict[str, Any]:
        """生成游戏故事情节"""
        try:
            # 输入验证
            if not self.current_task.get("prompt"):
                raise ValueError("缺少prompt参数")
                
            prompt = str(self.current_task["prompt"]).strip()[:500]  # 增加输入长度限制
            story_type = self.current_task.get("story_type", "fantasy")
            background = self.current_task.get("background", "")
            characters = self.current_task.get("characters", [])
            branch_points = self.current_task.get("branch_points", [])
            custom_type_desc = self.current_task.get("custom_type_desc", "")
            
            # 支持自定义故事类型
            type_descriptions = {
                "fantasy": "西方奇幻风格，包含魔法、龙等元素",
                "sci-fi": "科幻风格，包含未来科技、外星文明",
                "wuxia": "武侠风格，包含门派、武功、江湖恩怨",
                "horror": "恐怖风格，包含悬疑、惊悚元素", 
                "custom": "自定义风格，根据用户输入生成",
            }
            
            # 构建角色描述
            chars_desc = ""
            if characters:
                chars_desc = "\n已有角色设定:\n" + "\n".join(
                    f"- {char['name']}: {char['desc']}" 
                    for char in characters
                )
            
            # 验证故事类型
            if story_type not in type_descriptions:
                raise ValueError(f"不支持的故事类型: {story_type}")
            
            story_desc = type_descriptions.get(story_type)
            if story_type == 'custom' and custom_type_desc:
                story_desc = custom_type_desc
                
            messages = [{
                "role": "system",
                "content": f"""你是一个专业的游戏故事生成器，擅长创作{story_desc}风格的故事情节。
                请根据用户需求生成完整的游戏故事，包含以下要素：
                - 详细的世界观设定（地理、历史、文化等）
                - 有深度的角色发展（背景故事、性格特点、成长弧线）
                - 多分支剧情设计（主线+3-5条支线）
                - 游戏化适配元素（任务设计、关卡机制、奖励系统）
                - 冲突与转折点设计（至少3个关键转折）
                - 结局多样性（至少2种不同结局）"""
            }, {
                "role": "user",
                "content": f"""游戏故事生成需求：
主题：{prompt}
故事类型：{story_type}
{'' if not background else '背景设定：' + background}
{chars_desc}

请生成包含以下内容的故事：
1. 世界观设定（200-300字）
2. 主要角色（3-5个，每个角色包含：
   - 背景故事
   - 性格特点
   - 角色成长弧线
3. 主线剧情（包含3-5个关键情节点）
4. 支线任务设计（2-3个）
5. 游戏化适配建议（如关卡设计、玩法机制等）"""
            }]
            
            try:
                print("调用DashScope API生成故事...")
                response = dashscope.Generation.call(
                    model='qwen-max',
                    messages=messages,
                    temperature=0.9,
                    top_p=0.95,
                    max_tokens=1200,
                    result_format='message',
                    seed=int(time.time())
                )
                
                print(f"API响应状态码: {response.status_code}")
                print(f"API响应内容: {response}")
                
                if response.status_code != HTTPStatus.OK:
                    print(f"API调用失败: {response.message}")
                    return {
                        "error": f"API调用失败: {response.message}",
                        "status": "failed"
                    }
                    
                if not hasattr(response, 'output') or not hasattr(response.output, 'choices'):
                    print("API返回格式异常")
                    return {
                        "error": "API返回格式异常",
                        "status": "failed"
                    }
                    
                content = response.output.choices[0].message.content
                print(f"生成的故事内容长度: {len(content)}")
                
                if not content.strip():
                    return {
                        "error": "API返回空内容",
                        "status": "failed"
                    }
                    
                return {
                    "story": content,
                    "status": "completed"
                }
            except Exception as e:
                print(f"故事生成过程中发生异常: {str(e)}")
                return {
                    "error": str(e),
                    "status": "failed"
                }
                    
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }
