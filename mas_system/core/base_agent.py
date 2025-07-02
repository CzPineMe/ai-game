from typing import Any, Dict
from .controller import CentralController

class BaseAgent:
    def __init__(self, agent_id: str, controller: CentralController):
        self.agent_id = agent_id
        self.controller = controller
        self.current_task = None
        self.register_with_controller()
        
    def register_with_controller(self):
        """向中央控制器注册"""
        self.controller.register_agent(
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__
        )
        
    def receive_task(self, task: Dict[str, Any]):
        """接收来自控制器的任务"""
        self.current_task = task
        self.controller.update_agent_status(
            agent_id=self.agent_id,
            status="working"
        )
        
    def complete_task(self, result: Dict[str, Any]):
        """完成任务并返回结果"""
        self.current_task = None
        self.controller.update_agent_status(
            agent_id=self.agent_id,
            status="idle"
        )
        return result
        
    def process_task(self):
        """处理任务的具体实现（由子类实现）"""
        raise NotImplementedError
