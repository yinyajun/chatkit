import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal

from chatkit.llm.base_client import Client, Metadata, StreamEvent, StreamEventType
from chatkit.llm.base_context import Context
from chatkit.llm.base_messages import Messages, ToolCall
from chatkit.logger import get_logger
from chatkit.services.anthropic.messages import AnthropicDynamicMessages, AnthropicStaticMessages
from chatkit.services.anthropic.params import AnthropicParams
from chatkit.services.anthropic.tools import AnthropicTool

logger = get_logger()

try:
    import anthropic
    from anthropic.types import CacheControlEphemeralParam

except ModuleNotFoundError as e:
    logger.log_error(f"Exception: {e}")
    logger.log_error("In order to use Anthropic, you need to `pip install echoflow[anthropic]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AnthropicContext(Context):
    params: AnthropicParams = field(default_factory=AnthropicParams)
    system: list[AnthropicStaticMessages] = field(default_factory=list)
    history: list[AnthropicStaticMessages] = field(default_factory=list)
    tools: list[AnthropicTool] = field(default_factory=list)


@dataclass
class AnthropicBedrockAuth:
    aws_region: str = None,
    aws_secret_key: str = None
    aws_access_key: str = None


@dataclass
class AnthropicRawAuth:
    api_key: str = None


class AnthropicClient(Client):
    def __init__(
            self,
            provider: Literal["anthropic", "bedrock", "vertex"] = "anthropic",
            auth_bedrock: AnthropicBedrockAuth = None,
            auth_raw: AnthropicRawAuth = None,
    ):
        if provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=auth_raw.api_key)

        elif provider == "bedrock":
            self.client = anthropic.AsyncAnthropicBedrock(
                aws_region=auth_bedrock.aws_region,
                aws_secret_key=auth_bedrock.aws_secret_key,
                aws_access_key=auth_bedrock.aws_access_key
            )

        elif provider == "vertex":
            self.client = anthropic.AsyncAnthropicVertex()  # todo

        else:
            raise ValueError("unsupported provider for AnthropicClient")

    @staticmethod
    def format_prompt(prompts: list[Messages]):
        messages = []
        has_dynamic = False

        for messages in prompts:
            if isinstance(messages, AnthropicStaticMessages):
                messages.extend(messages.value)

            elif isinstance(messages, AnthropicDynamicMessages):
                if not has_dynamic and len(messages) > 0:
                    messages[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
                has_dynamic = True
                messages.extend(messages.value)

        if not has_dynamic and len(messages) > 0:
            messages[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        return messages

    async def stream_generate(self, ctx: AnthropicContext, **kwargs) -> AsyncGenerator[StreamEvent, None]:
        params = ctx.params
        tools = [t.marshal() for t in ctx.tools]
        system = self.format_prompt(ctx.system)
        history = self.format_prompt(ctx.history)

        stream = await self.client.messages.create(
            tools=tools,
            messages=history,
            model=params.model_id,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            max_tokens=params.max_tokens,
            system=system,
            stream=True,
        )
        async for event in self._process_stream(stream):
            yield event

    async def _process_stream(self, stream) -> AsyncGenerator[StreamEvent, None]:
        tool_id = None
        tool_name = None
        tool_arguments = ""
        meta = Metadata()

        async for event in stream:
            logger.log_debug(f"[AnthropicClient Stream Event] {event}")

            if event.type == "message_start":
                message = event.message
                usage = message.usage
                meta.cache_read_tokens = usage.cache_read_input_tokens
                meta.cache_write_tokens = usage.cache_creation_input_tokens
                meta.input_text_tokens = usage.input_tokens

                yield StreamEvent(type=StreamEventType.start, data={})
                continue

            elif event.type == "content_block_start":
                content_block = event.content_block
                if content_block.type == "tool_use":
                    tool_id = content_block.id
                    tool_name = content_block.name

            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    text_delta = delta.text
                    if not text_delta:
                        continue

                    yield StreamEvent(
                        type=StreamEventType.text_delta, data={"text_delta": text_delta}
                    )

                elif delta.type == "input_json_delta":
                    tool_arguments += delta.partial_json

            elif event.type == "content_block_stop":
                if tool_id:
                    try:
                        arguments = json.loads(tool_arguments)
                    except:
                        arguments = {}

                    tool_call_info = ToolCall(id=tool_id, name=tool_name, input=arguments)
                    yield StreamEvent(type=StreamEventType.tool, data={"tool": tool_call_info})

                    tool_id = None
                    tool_name = None
                    tool_arguments = ""

            elif event.type == "message_delta":
                stop_reason = event.delta.stop_reason
                yield StreamEvent(type=StreamEventType.stop, data={"stop_reason": stop_reason})

                usage = event.usage
                meta.output_text_tokens = usage.output_tokens
                yield StreamEvent(type=StreamEventType.metadata, data={"metadata": meta})
