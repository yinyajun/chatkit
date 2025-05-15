from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

from chatkit.llm.base_messages import ToolCall, ToolResult
from chatkit.llm.json_schema import Model


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    id: str
    content: str


@dataclass
class Tool:
    name: str
    """The name of the tool, used for identification."""
    description: str
    """A description of the tool, explaining its functionality and purpose."""
    input_schema: Type[Model]
    """The schema defining the structure of the input data required by the tool."""

    @abstractmethod
    def marshal(self):
        pass

    def copy(self) -> "Tool":
        return Tool(self.name, self.description, self.input_schema)

    @abstractmethod
    async def async_call(self, tool_call: ToolCall) -> ToolResult:
        """Asynchronously call the tool.

        Args:
            tool_call (ToolCall): The input information for the tool call.

        Returns:
            ToolResult: The result of the tool call.
        """
        pass

    @abstractmethod
    def call(self, tool_call: ToolCall) -> ToolResult:
        """Synchronously call the tool.

        Args:
            tool_call (ToolCall): The input information for the tool call.

        Returns:
            ToolResult: The result of the tool call.
        """
        pass


class ToolWrapper(Tool):
    def __init__(self, tool: Tool):
        self._source = tool
        self.__dict__.update(self._source.__dict__)

    def copy(self) -> "Tool":
        return self._source.copy()

    def call(self, tool_call: ToolCall) -> ToolResult:
        return self._source.call(tool_call)

    async def async_call(self, tool_call: ToolCall) -> ToolResult:
        return await self._source.async_call(tool_call)
