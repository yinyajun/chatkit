import asyncio
import os
import unittest

from echoflow.llm.base_client import StreamEvent, StreamEventType
from echoflow.llm.base_messages import Message, ToolCall, ToolResult
from echoflow.services.anthropic.client import AnthropicClient, AnthropicContext, AnthropicRawAuth
from echoflow.services.anthropic.messages import (
    AnthropicDynamicMessages,
    AnthropicStaticMessages,
)


class TestAnthropicClientWithRaw(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = AnthropicClient(auth_raw=AnthropicRawAuth(api_key=os.getenv("anthropic_api_key")))
        self.ctx = AnthropicContext()

    async def test_stream_generate(self):
        events = []
        async for event in self.client.stream_generate(self.ctx):
            events.append(event)

        self.assertTrue(events, "No events were generated")
        has_start = any(event.type == StreamEventType.start for event in events)
        has_metadata = any(event.type == StreamEventType.metadata for event in events)
        has_stop = any(event.type == StreamEventType.stop for event in events)

        self.assertTrue(has_start, "Start event is missing")
        self.assertTrue(has_metadata, "Metadata event is missing")
        self.assertTrue(has_stop, "Stop event is missing")


async def run():
    client = AnthropicClient(auth_raw=AnthropicRawAuth(api_key=os.getenv("anthropic_api_key")))
    history = AnthropicStaticMessages()
    history.add_message(Message(role="user", content=["hello hello hello"]))
    ctx = AnthropicContext(history=history)
    async for event in client.stream_generate(ctx):
        print(777, event)


asyncio.run(run())

# 1111111111111
# 1111111111111
# 2025-05-03 08:47:46.538 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawMessageStartEvent(message=Message(id='msg_011CN5HDeXovxZxZYWrkRcXV', content=[], model='claude-3-5-sonnet-20241022', role='assistant', stop_reason=None, stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=9, output_tokens=1)), type='message_start')
# 777 StreamEvent(type=<StreamEventType.start: 1>, data={})
# 2025-05-03 08:47:46.538 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawContentBlockStartEvent(content_block=TextBlock(citations=None, text='', type='text'), index=0, type='content_block_start')
# 2025-05-03 08:47:46.538 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawContentBlockDeltaEvent(delta=TextDelta(text='Hi', type='text_delta'), index=0, type='content_block_delta')
# 777 StreamEvent(type=<StreamEventType.text_delta: 2>, data={'text_delta': 'Hi'})
# 2025-05-03 08:47:46.649 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawContentBlockDeltaEvent(delta=TextDelta(text='! How can I help', type='text_delta'), index=0, type='content_block_delta')
# 777 StreamEvent(type=<StreamEventType.text_delta: 2>, data={'text_delta': '! How can I help'})
# 2025-05-03 08:47:46.735 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawContentBlockDeltaEvent(delta=TextDelta(text=' you today?', type='text_delta'), index=0, type='content_block_delta')
# 777 StreamEvent(type=<StreamEventType.text_delta: 2>, data={'text_delta': ' you today?'})
# 2025-05-03 08:47:46.736 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawContentBlockStopEvent(index=0, type='content_block_stop')
# 2025-05-03 08:47:46.739 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawMessageDeltaEvent(delta=Delta(stop_reason='end_turn', stop_sequence=None), type='message_delta', usage=MessageDeltaUsage(output_tokens=12))
# 777 StreamEvent(type=<StreamEventType.stop: 7>, data={'stop_reason': 'end_turn'})
# 777 StreamEvent(type=<StreamEventType.metadata: 6>, data={'metadata': {'input_usage': Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=9, output_tokens=1), 'output_usage': MessageDeltaUsage(output_tokens=12)}})
# 2025-05-03 08:47:46.739 | DEBUG    | echoflow.logger.logger:log_debug:28 - A#### RawMessageStopEvent(type='message_stop')
