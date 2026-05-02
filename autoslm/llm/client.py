"""Provider-agnostic LLM client via LiteLLM.

Supports Anthropic (default, paper), Gemini, OpenAI, DeepSeek, local OpenAI-compatible.
Model strings use LiteLLM convention: "anthropic/claude-sonnet-4-6", "gemini/gemini-2.5-pro",
"openai/gpt-5", "deepseek/deepseek-reasoner", "openai/<local>" with custom base_url.

Prompt caching is enabled automatically for Anthropic via cache_control breakpoints
when system prompt or tools are large (paper runs 500-1500 turns; caching is critical).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import json
import os

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    raw: Any = None
    usage: dict = field(default_factory=dict)
    stop_reason: Optional[str] = None


@dataclass
class LLMClient:
    model: str
    temperature: float = 0.7
    max_tokens: int = 8192
    thinking_budget: int = 0  # >0 enables Anthropic extended thinking
    cache_system: bool = True
    api_base: Optional[str] = None

    def _is_anthropic(self) -> bool:
        return self.model.startswith("anthropic/") or "claude" in self.model.lower()

    def _build_messages(self, system: Optional[str], messages: list[dict]) -> tuple[Any, list[dict]]:
        sys_block: Any = None
        if system:
            if self._is_anthropic() and self.cache_system:
                sys_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                sys_block = system
        return sys_block, messages

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30))
    def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        import litellm  # lazy import

        sys_block, msgs = self._build_messages(system, messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if sys_block is not None:
            kwargs["system"] = sys_block if self._is_anthropic() else None
            if not self._is_anthropic():
                kwargs["messages"] = [{"role": "system", "content": system}] + msgs
                kwargs.pop("system", None)
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if self.thinking_budget and self._is_anthropic():
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        if self.api_base:
            kwargs["api_base"] = self.api_base

        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        text = getattr(msg, "content", "") or ""
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict) and b.get("type") == "text")

        tcs = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            tcs.append({"id": tc.id, "name": tc.function.name, "arguments": args})

        return LLMResponse(
            content=text,
            tool_calls=tcs,
            raw=resp,
            usage=getattr(resp, "usage", {}) or {},
            stop_reason=resp.choices[0].finish_reason,
        )


def chat(model: str, prompt: str, system: Optional[str] = None, **kw) -> str:
    return LLMClient(model=model, **kw).complete(
        [{"role": "user", "content": prompt}], system=system
    ).content


def chat_with_tools(
    client: LLMClient,
    messages: list[dict],
    tools: list[dict],
    tool_handlers: dict[str, Callable[[dict], Any]],
    system: Optional[str] = None,
    max_iters: int = 50,
    on_step: Optional[Callable[[int, LLMResponse], None]] = None,
) -> list[dict]:
    """Tool-use loop. Returns full conversation. Stops on no tool_calls or max_iters."""
    convo = list(messages)
    for i in range(max_iters):
        resp = client.complete(convo, system=system, tools=tools)
        if on_step:
            on_step(i, resp)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in resp.tool_calls
            ]
        convo.append(assistant_msg)
        if not resp.tool_calls:
            return convo
        for tc in resp.tool_calls:
            handler = tool_handlers.get(tc["name"])
            if handler is None:
                result = {"error": f"unknown tool {tc['name']}"}
            else:
                try:
                    result = handler(tc["arguments"] or {})
                except Exception as e:
                    result = {"error": str(e)}
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, default=str)[:50_000],
                }
            )
    return convo
