import unittest

from echoflow.llm.base_messages import Message, ToolCall, ToolResult
from echoflow.services.anthropic.messages import (
    AnthropicAdapter,
    AnthropicDynamicMessages,
    AnthropicStaticMessages,
)


class TestAnthropicAdapter(unittest.TestCase):
    def setUp(self):
        self.adaptor = AnthropicAdapter()

    def test_adapt(self):
        message = Message(role="user", content=["request?"])
        expected = {"role": "user", "content": [{"text": "request?", "type": "text"}]}
        self.assertEqual(self.adaptor.adapt(message), expected)

    def test_to_message(self):
        element = {"role": "user", "content": [{"text": "request?", "type": "text"}]}
        expected = Message(role="user", content=["request?"])
        self.assertEqual(self.adaptor.to_message(element), expected)

        element = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "12345", "content": "54321"}],
        }
        expected = Message(role="tool", content=[ToolResult(id="12345", content="54321")])
        self.assertEqual(self.adaptor.to_message(element), expected)

        element = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "12345", "name": "random", "input": {"args": "12345"}}
            ],
        }
        expected = Message(
            role="assistant", content=[ToolCall(id="12345", name="random", input={"args": "12345"})]
        )
        self.assertEqual(self.adaptor.to_message(element), expected)


class TestAnthropicDynamicMessages(unittest.TestCase):
    def setUp(self):
        messages = AnthropicDynamicMessages()
        m1 = Message(role="user", content=["first request"])
        m2 = Message(role="user", content=["second request"])
        m3 = Message(role="user", content=["what request?"])
        messages.add_message(m1)
        messages.add_message(m2)
        messages.add_message(Message(role="assistant", content=["first reply"]))
        messages.add_message(
            Message(
                role="assistant",
                content=[
                    "second reply",
                    ToolCall(id="12345", name="random", input={"args": "12345"}),
                ],
            )
        )
        messages.add_message(
            Message(role="tool", content=[ToolResult(id="12345", content="54321")])
        )
        messages.add_message(m3)
        messages.add_message(Message(role="assistant", content=["yes"]))
        self.messages = messages

        # dynamic modify message content
        m1.content[0] = "1st request"
        m3.content[0] = "which request?"

    def test_message(self):
        expected = [
            Message(role="user", content=["1st request"]),
            Message(role="user", content=["second request"]),
            Message(role="assistant", content=["first reply"]),
            Message(
                role="assistant",
                content=[
                    "second reply",
                    ToolCall(id="12345", name="random", input={"args": "12345"}),
                ],
            ),
            Message(role="tool", content=[ToolResult(id="12345", content="54321")]),
            Message(role="user", content=["which request?"]),
            Message(role="assistant", content=["yes"]),
        ]
        self.assertEqual(self.messages, expected)

    def test_message_value(self):
        expected = [
            {
                "role": "user",
                "content": [
                    {"text": "1st request", "type": "text"},
                    {"text": "second request", "type": "text"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "first reply", "type": "text"},
                    {"text": "second reply", "type": "text"},
                    {
                        "type": "tool_use",
                        "id": "12345",
                        "name": "random",
                        "input": {"args": "12345"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "12345", "content": "54321"}],
            },
            {"role": "user", "content": [{"text": "which request?", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "yes", "type": "text"}]},
        ]
        self.assertEqual(self.messages.value, expected)


class TestAnthropicStaticMessages(unittest.TestCase):
    def setUp(self):
        messages = AnthropicStaticMessages()
        messages.add_message(Message(role="user", content=["first request"]))
        messages.add_message(Message(role="user", content=["second request"]))
        messages.add_message(Message(role="assistant", content=["first reply"]))
        messages.add_message(
            Message(
                role="assistant",
                content=[
                    "second reply",
                    ToolCall(id="12345", name="random", input={"args": "12345"}),
                ],
            )
        )
        messages.add_message(
            Message(role="tool", content=[ToolResult(id="12345", content="54321")])
        )
        messages.add_message(Message(role="user", content=["what request?"]))
        messages.add_message(Message(role="assistant", content=["yes"]))
        self.messages = messages

    def test_message(self):
        expected = [
            Message(role="user", content=["first request", "second request"]),
            Message(
                role="assistant",
                content=[
                    "first reply",
                    "second reply",
                    ToolCall(id="12345", name="random", input={"args": "12345"}),
                ],
            ),
            Message(role="tool", content=[ToolResult(id="12345", content="54321")]),
            Message(role="user", content=["what request?"]),
            Message(role="assistant", content=["yes"]),
        ]
        self.assertEqual(self.messages, expected)

    def test_message_value(self):
        expected = [
            {
                "role": "user",
                "content": [
                    {"text": "first request", "type": "text"},
                    {"text": "second request", "type": "text"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "first reply", "type": "text"},
                    {"text": "second reply", "type": "text"},
                    {
                        "type": "tool_use",
                        "id": "12345",
                        "name": "random",
                        "input": {"args": "12345"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "12345", "content": "54321"}],
            },
            {"role": "user", "content": [{"text": "what request?", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "yes", "type": "text"}]},
        ]
        self.assertEqual(self.messages.value, expected)
