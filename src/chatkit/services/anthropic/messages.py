from typing import Literal

from chatkit.llm.base_messages import (
    DynamicMessages,
    Message,
    MessageAdapter,
    StaticMessages,
    Text,
    ToolCall,
    ToolResult,
)
from chatkit.logger import get_logger

logger = get_logger()

try:
    from anthropic.types import (
        MessageParam,
        TextBlockParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
    )
except ModuleNotFoundError as e:
    logger.log_error(f"Exception: {e}")
    logger.log_error("In order to use Anthropic, you need to `pip install echoflow[anthropic]`.")
    raise Exception(f"Missing module: {e}")


class AnthropicAdapter(MessageAdapter):
    def adapt(self, message: Message):
        role: Literal["user", "assistant"] = "user"
        if message.role == "assistant":
            role = "assistant"
        elif message.role == "tool":
            role = "user"

        content = []
        for c in message.content:
            if type(c) == Text:
                content.append(TextBlockParam(text=c, type="text"))
            elif type(c) == ToolCall:
                content.append(
                    ToolUseBlockParam(type="tool_use", id=c.id, name=c.name, input=c.input)
                )
            elif type(c) == ToolResult:
                content.append(
                    ToolResultBlockParam(type="tool_result", tool_use_id=c.id, content=c.content)
                )

        return MessageParam(role=role, content=content)

    def to_message(self, element) -> Message:
        role: Literal["user", "assistant", "tool"] = "user"
        if element["role"] == "assistant":
            role = "assistant"

        content = []
        for c in element["content"]:
            if c["type"] == "text":
                content.append(c["text"])
            elif c["type"] == "tool_use":
                content.append(ToolCall(id=c["id"], name=c["name"], input=c["input"]))
            elif c["type"] == "tool_result":
                role = "tool"
                content.append(ToolResult(id=c["tool_use_id"], content=c["content"]))

        return Message(role=role, content=content)


class AnthropicStaticMessages(StaticMessages):
    def __init__(self):
        super().__init__(adapter=AnthropicAdapter())


class AnthropicDynamicMessages(DynamicMessages):
    def __init__(self):
        super().__init__(adapter=AnthropicAdapter())
