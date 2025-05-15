from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncGenerator, Optional

from chatkit.llm.base_context import LLMContext
from chatkit.llm.base_messages import ToolCall


class StreamEventType(Enum):
    start = auto()
    text_delta = auto()
    thinking_delta = auto()
    tool = auto()
    error = auto()
    metadata = auto()
    stop = auto()


@dataclass
class StreamEvent:
    type: StreamEventType
    data: dict = field(default_factory=dict)


@dataclass
class Metadata:
    cache_write_tokens: Optional[int] = 0
    cache_read_tokens: Optional[int] = 0
    input_text_tokens: Optional[int] = 0
    output_text_tokens: Optional[int] = 0


@dataclass
class Result:
    text: str = ""
    tool_call: Optional[ToolCall] = None
    metadata: Optional[Metadata] = None


class Client:
    @abstractmethod
    async def stream_generate(self, ctx: LLMContext, **kwargs) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplemented()

    async def generate(self, ctx: LLMContext, **kwargs) -> Result:
        text = ""
        tool_call: Optional[ToolCall] = None
        metadata: Optional[Metadata] = None
        async for event in self.stream_generate(ctx, **kwargs):
            if event.type == StreamEventType.text_delta:
                text += event.data["text"]

            elif event.type == StreamEventType.tool:
                tool_call = event.data["tool"]

            elif event.type == StreamEventType.metadata:
                metadata = event.data["metadata"]

        return Result(text=text, tool_call=tool_call, metadata=metadata)

