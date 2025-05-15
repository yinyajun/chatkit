from dataclasses import dataclass, field

from chatkit.llm.base_messages import Messages
from chatkit.llm.base_tools import Tool


@dataclass
class Params:
    model_id: str
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.8


@dataclass
class Context:
    params: Params = field(default_factory=Params)
    system: list[Messages] = field(default_factory=list)  # 路由问题？交给上层
    history: list[Messages] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)  # 路由问题？交给上层
