from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AgentInfo:
    agent_id: str
    agent_type: str
    status: str = "idle"

class CentralController:
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.task_queue: List[dict] = []
        
    def register_agent(self, agent_id: str, agent_type: str):
        """注册新智能体"""
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type
        )
        
    def dispatch_task(self, task: dict):
        """分配任务给合适的智能体"""
        self.task_queue.append(task)
        
    def get_agent_status(self, agent_id: str) -> str:
        """获取智能体状态"""
        return self.agents[agent_id].status
        
    def update_agent_status(self, agent_id: str, status: str):
        """更新智能体状态"""
        self.agents[agent_id].status = status
