from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, List, Literal, Union

from chatkit.llm.base_tools import ToolCall, ToolResult

Text = str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: List[Union[Text, ToolCall, ToolResult]]


class MessageAdapter(ABC):
    @abstractmethod
    def adapt(self, message: Message) -> Any:
        pass

    @abstractmethod
    def to_message(self, element) -> Message:
        pass


class Messages(ABC, list):
    @abstractmethod
    def add_message(self, message: Message):
        pass

    @property
    @abstractmethod
    def value(self) -> list:
        pass


class StaticMessages(Messages):
    def __init__(self, adapter: MessageAdapter = None):
        super().__init__()
        self.adapter = adapter
        self._value = []

    def _alternate_role(self, role: Literal["user", "assistant", "tool"]) -> bool:
        if len(self) == 0 or self[-1].role != role:
            self.append(Message(role=role, content=[]))
            self._value.append(Message(role=role, content=[]))
            return True
        return False

    def _add_item(
            self,
            role: Literal["user", "assistant", "tool"],
            item: Union[str, ToolCall, ToolResult]
    ):
        if not item:
            return

        self._alternate_role(role)
        self[-1].content.append(item)
        self._value[-1] = self.adapter.adapt(self[-1]) if self.adapter else self[-1]

    def add_message(self, message: Message):
        match message.role:
            case "user" | "assistant":
                for i, content in enumerate(message.content):
                    if isinstance(content, Text):
                        self._add_item(message.role, content)
                    elif isinstance(content, ToolCall):
                        self._add_item("assistant", content)
                    else:
                        raise TypeError(f"{type(content)} content is not supported")

            case "tool":
                tool_result = message.content[0]
                assert isinstance(tool_result, ToolResult), "Expected type is <ToolResult>"
                self._add_item("tool", tool_result)

            case _:
                raise ValueError(f"role {message.role} is not supported")

    @property
    def value(self) -> list:
        return self._value


class DynamicMessages(Messages):
    def __init__(self, adapter: MessageAdapter = None):
        super().__init__()
        self.adapter = adapter

    def add_message(self, message: Message):
        self.append(message)

    @property
    def value(self) -> list:
        res = StaticMessages(self.adapter)

        for i, m in enumerate(self):
            res.add_message(m)

        return res.value
