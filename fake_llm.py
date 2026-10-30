"""测试用假模型：模拟 ChatAnthropic.with_structured_output(schema, method=...).invoke(messages)。

按 schema 指定返回值：FakeLLM({RouterResult: RouterResult(intent="qa"), VerifyResult: RuntimeError()})
值为异常实例时抛出；值为列表时按调用顺序依次返回（用完后重复最后一个）；
未指定的 schema 被调用时抛 AssertionError（节点会将其当作模型出错处理）。
"""

from typing import Any


class FakeLLM:
    def __init__(self, responses: dict[type, Any]):
        self.responses = responses
        self.calls: list[tuple[type, Any, list]] = []  # (schema, method, messages)

    def with_structured_output(self, schema, method=None):
        return _Bound(self, schema, method)

    def calls_for(self, schema) -> list[list]:
        return [messages for s, _, messages in self.calls if s is schema]


class _Bound:
    def __init__(self, llm: FakeLLM, schema, method):
        self.llm, self.schema, self.method = llm, schema, method

    def invoke(self, messages):
        self.llm.calls.append((self.schema, self.method, messages))
        if self.schema not in self.llm.responses:
            raise AssertionError(f"unexpected call for {self.schema.__name__}")
        value = self.llm.responses[self.schema]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, BaseException):
            raise value
        return value
