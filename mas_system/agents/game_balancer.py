from typing import Dict, Any, List, Optional
from datetime import datetime
from ..core.base_agent import BaseAgent
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

class GameBalancerAgent(BaseAgent):
    def __init__(self, agent_id: str, controller):
        super().__init__(agent_id, controller)
        self.player_data = pd.DataFrame()
        self.real_time_data = []
        self.last_analysis_time = None
        self.adjustment_history = []
        
    def process_task(self):
        """处理游戏平衡任务"""
        if not self.current_task:
            raise ValueError("没有当前任务")
            
        task_type = self.current_task.get("type")
        
        if task_type == "analyze_data":
            return self.analyze_player_data()
        elif task_type == "adjust_balance":
            return self.provide_adjustments()
        elif task_type == "real_time_analysis":
            return self.real_time_analysis()
        elif task_type == "get_adjustment_history":
            return self.get_adjustment_history()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
            
    def real_time_analysis(self) -> Dict[str, Any]:
        """实时分析玩家行为数据"""
        if not self.current_task.get("player_data"):
            return {"status": "error", "message": "缺少玩家数据"}
            
        # 添加时间戳
        data = self.current_task["player_data"]
        data["timestamp"] = datetime.now().isoformat()
        self.real_time_data.append(data)
        
        # 每10条数据执行一次分析
        if len(self.real_time_data) >= 10:
            analysis = self._analyze_real_time_data()
            self.last_analysis_time = datetime.now()
            return {
                "status": "completed",
                "analysis": analysis,
                "suggestions": self._generate_real_time_suggestions(analysis)
            }
        return {"status": "pending", "message": "等待更多数据"}
        
    def get_adjustment_history(self) -> Dict[str, Any]:
        """获取调整历史记录"""
        return {
            "status": "completed",
            "history": self.adjustment_history
        }
        
    def _analyze_real_time_data(self) -> Dict[str, Any]:
        """分析实时数据"""
        df = pd.DataFrame(self.real_time_data)
        self.real_time_data = []  # 清空缓存
        
        # 标准化数据
        scaler = StandardScaler()
        features = df[["completion_time", "attempts", "success"]]
        scaled_features = scaler.fit_transform(features)
        
        # 异常检测
        clf = IsolationForest(contamination=0.1)
        df["anomaly"] = clf.fit_predict(scaled_features)
        
        return {
            "average_completion_time": df["completion_time"].mean(),
            "success_rate": df["success"].mean(),
            "anomalies": df[df["anomaly"] == -1].to_dict("records"),
            "data_points": len(df)
        }
        
    def _generate_real_time_suggestions(self, analysis: Dict) -> List[str]:
        """生成实时调整建议"""
        suggestions = []
        
        if analysis["success_rate"] < 0.4:
            suggestions.append("建议降低当前关卡难度")
        if analysis["average_completion_time"] > 300:
            suggestions.append("建议优化关卡流程设计")
        if analysis["anomalies"]:
            suggestions.append(f"检测到{len(analysis['anomalies'])}个异常数据点，建议检查")
            
        # 记录调整建议
        if suggestions:
            self.adjustment_history.append({
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions,
                "analysis": analysis
            })
            
        return suggestions
            
    def analyze_player_data(self) -> Dict[str, Any]:
        """分析玩家行为数据"""
        data = self.current_task["player_data"]
        self.player_data = pd.DataFrame(data)
        
        # 分析关键指标
        analysis = {
            "completion_rate": self._calculate_completion_rate(),
            "difficulty_clusters": self._cluster_difficulty_levels(),
            "hotspots": self._identify_hotspots()
        }
        
        return {
            "analysis": analysis,
            "status": "completed"
        }
        
    def provide_adjustments(self) -> Dict[str, Any]:
        """提供游戏平衡调整建议"""
        if self.player_data.empty:
            raise ValueError("没有可用的玩家数据")
            
        suggestions = []
        # 根据分析结果生成建议
        if self._calculate_completion_rate() < 0.5:
            suggestions.append("降低关卡难度")
        if len(self._cluster_difficulty_levels()) > 3:
            suggestions.append("优化难度曲线")
            
        return {
            "suggestions": suggestions,
            "status": "completed"
        }
        
    def _calculate_completion_rate(self) -> float:
        """计算关卡完成率"""
        if "success" not in self.player_data.columns:
            return 0.0
        return self.player_data["success"].mean()
        
    def _cluster_difficulty_levels(self) -> List[int]:
        """聚类分析难度级别"""
        if "completion_time" not in self.player_data.columns:
            return []
            
        X = self.player_data[["completion_time", "attempts"]].values
        kmeans = KMeans(n_clusters=3).fit(X)
        return kmeans.labels_.tolist()
        
    def _identify_hotspots(self) -> Dict[str, float]:
        """识别玩家卡点"""
        if "fail_location" not in self.player_data.columns:
            return {}
            
        return self.player_data["fail_location"].value_counts().to_dict()
