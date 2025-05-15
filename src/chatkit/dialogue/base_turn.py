from typing import Optional, List
from dataclasses import dataclass, field

from chatkit.llm.base_tools import ToolCall, ToolResult


@dataclass
class Request:
    stream_id: str
    utterance: str = ""


@dataclass
class Reply:
    utterance: str = ""
    """
    The text content of the reply. 
    """
    tool_call: ToolCall = None
    tool_result: ToolResult = None
    played_utterance: str = ""
    """
    The text that has been played. 
    """
    interrupted: bool = False
    """
    Indicates whether the reply was interrupted. 
    """


@dataclass
class Turn:
    id: str = ""
    requests: list[Request] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    reply_idx: int = -1
    play_idx: int = -1

    def add_user_utterance(self, stream_id: str, text: str):
        self.requests.append(Request(stream_id=stream_id, utterance=text))
        self.id = stream_id

    def add_reply(self,
                  stream_id: str,
                  text: str,
                  tool_call_info: ToolCall,
                  tool_call_result: ToolResult):
        if stream_id != self.id:
            return

        self.reply_idx += 1
        if self.reply_idx < len(self.replies):
            self.replies[self.reply_idx].text = text
            self.replies[self.reply_idx].tool_call = tool_call_info
            self.replies[self.reply_idx].tool_result = tool_call_result
        else:
            reply = Reply(
                text=text,
                tool_call=tool_call_info,
                tool_result=tool_call_result
            )
            self.replies.append(reply)
        return self.replies[self.reply_idx]
