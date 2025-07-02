from typing import Dict, Any
import os
import json
from ..core.base_agent import BaseAgent
import dashscope
from http import HTTPStatus

class NPCAgent(BaseAgent):
    def __init__(self, agent_id: str, controller):
        super().__init__(agent_id, controller)
        self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.dashscope_key:
            raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
        self.dialogue_history = []  # 对话历史记录
        self.personality = "友好且乐于助人"  # NPC默认性格
        self.history_file = f"data/npc_dialogues_{agent_id}.json"
        self.load_dialogue_history()
        
    def process_task(self):
        """处理NPC行为任务"""
        if not self.current_task:
            raise ValueError("没有当前任务")
            
        task_type = self.current_task.get("type")
        
        if task_type == "dialogue":
            return self.generate_dialogue()
        elif task_type == "emotional_response":
            return self.generate_emotional_response()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
            
    def load_dialogue_history(self):
        """从文件加载对话历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.dialogue_history = json.load(f)
        except Exception as e:
            print(f"加载对话历史失败: {e}")

    def save_dialogue_history(self):
        """保存对话历史到文件"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialogue_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话历史失败: {e}")

    def clear_dialogue_history(self):
        """清空对话历史"""
        self.dialogue_history = []
        self.save_dialogue_history()
        return {"status": "completed", "message": "对话历史已清空"}

    def generate_dialogue(self) -> Dict[str, str]:
        """生成NPC对话"""
        context = self.current_task["context"]
        
        # 构建对话历史
        messages = [{
            "role": "system",
            "content": f"你是一个游戏NPC，性格特点：{self.personality}。需要根据对话上下文生成自然的回应"
        }]
        
        # 添加历史对话
        for dialogue in self.dialogue_history[-5:]:  # 保留最近5轮对话
            messages.append({
                "role": "user",
                "content": dialogue["player_input"]
            })
            messages.append({
                "role": "assistant",
                "content": dialogue["npc_response"]
            })
            
        # 添加当前对话
        messages.append({
            "role": "user",
            "content": context
        })
        
        dashscope.api_key = self.dashscope_key
        response = dashscope.Generation.call(
            model='qwen-max',
            messages=messages,
            temperature=0.7,
            result_format='message'
        )
        
        if response.status_code != HTTPStatus.OK:
            return {
                "error": f"API调用失败: {response.message}",
                "status": "failed"
            }
            
        npc_response = response.output.choices[0].message.content
        
        # 记录对话历史
        self.dialogue_history.append({
            "player_input": context,
            "npc_response": npc_response
        })
        
        # 保存对话历史
        self.save_dialogue_history()
        
        # 根据情感添加表情符号
        emotion_icons = {
            "positive": "😊",
            "neutral": "😐", 
            "negative": "😢"
        }
        sentiment = self.analyze_sentiment(npc_response)
        icon = emotion_icons.get(sentiment["label"], "💬")
        
        return {
            "dialogue": f"{icon} {npc_response}",
            "status": "completed",
            "sentiment": sentiment
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        response = dashscope.Generation.call(
            model='qwen-max',
            messages=[{
                "role": "system",
                "content": "分析以下文本的情感倾向，返回label(positive/neutral/negative)和score(0-1)"
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.3
        )
        
        if response.status_code != HTTPStatus.OK:
            return {"label": "neutral", "score": 0.5}
        try:
            return eval(response.output.choices[0].message.content)
        except:
            return {"label": "neutral", "score": 0.5}

    def generate_emotional_response(self) -> Dict[str, Any]:
        """根据玩家情感生成响应"""
        player_input = self.current_task["player_input"]
        
        # 记录玩家输入
        self.dialogue_history.append({
            "player_input": player_input,
            "npc_response": None
        })
        
        # 情感分析
        sentiment_response = dashscope.Generation.call(
            model='qwen-max',
            messages=[{
                "role": "system",
                "content": "分析以下文本的情感倾向，返回label(positive/neutral/negative)和score(0-1)"
            }, {
                "role": "user",
                "content": player_input
            }],
            temperature=0.3
        )
        
        if sentiment_response.status_code != HTTPStatus.OK:
            sentiment = {"label": "neutral", "score": 0.5}
        else:
            try:
                sentiment = eval(sentiment_response.output.choices[0].message.content)
            except:
                sentiment = {"label": "neutral", "score": 0.5}

        # 根据情感生成响应
        emotion_map = {
            "positive": "友好热情",
            "neutral": "平静礼貌", 
            "negative": "谨慎安抚"
        }
        tone = emotion_map.get(sentiment["label"], "neutral")
        
        messages = [{
            "role": "system",
            "content": f"你是一个游戏NPC，玩家当前情感状态为{sentiment['label']}(置信度{sentiment['score']:.2f})，请以{tone}的语气回应"
        }, {
            "role": "user", 
            "content": player_input
        }]
        
        dashscope.api_key = self.dashscope_key
        response = dashscope.Generation.call(
            model='qwen-max',
            messages=messages,
            temperature=0.7,
            result_format='message'
        )
        
        if response.status_code != HTTPStatus.OK:
            return {
                "error": f"API调用失败: {response.message}",
                "status": "failed"
            }
            
        npc_response = response.output.choices[0].message.content
        
        # 更新最后一条记录的NPC响应
        if self.dialogue_history and self.dialogue_history[-1]["npc_response"] is None:
            self.dialogue_history[-1]["npc_response"] = npc_response
            
        return {
            "response": npc_response,
            "sentiment": sentiment,
            "status": "completed"
        }
