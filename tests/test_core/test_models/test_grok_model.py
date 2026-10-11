"""Tests for GrokModel settings + secret handling (GROK_* + TEMPERATURE)."""

from types import SimpleNamespace

import deepeval.models.llms.grok_model as grok_mod
import pytest

from pydantic import SecretStr

from deepeval.models.llms.grok_model import GrokModel
from deepeval.test_case import MLLMImage
from tests.test_core.stubs import _RecordingClient


def _stub_load_model(monkeypatch):
    """Avoid importing xai_sdk in tests by stubbing load_model."""

    def fake_load_model(self, async_mode: bool = False):
        return _RecordingClient()

    monkeypatch.setattr(
        grok_mod.GrokModel,
        "load_model",
        fake_load_model,
        raising=True,
    )


#####################################
# API key / model name / temperature
#####################################


def test_grok_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch, settings
):
    """
    Explicit ctor api_key must override Settings.GROK_API_KEY, and the client
    should see a plain string, even if Settings stores a SecretStr.
    """
    # Seed env so Settings sees a GROK_API_KEY
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "env-secret-key"
        settings.GROK_MODEL_NAME = "grok-3"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    # Sanity: Settings should expose this as SecretStr
    assert isinstance(settings.GROK_API_KEY, SecretStr)

    # Prevent __init__ from importing xai_sdk
    _stub_load_model(monkeypatch)

    # ctor api_key should win over Settings.GROK_API_KEY
    model = GrokModel(
        model="grok-3",
        api_key="ctor-secret-key",
    )

    # _build_client should unwrap the SecretStr to a plain string
    client = model._build_client(_RecordingClient)
    api_key = client.kwargs.get("api_key")

    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_grok_model_defaults_from_settings(monkeypatch, settings):
    """
    When no ctor args are provided, GrokModel should pull model/api_key from
    Settings, which are backed by env vars.
    """
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "env-secret-key"
        settings.GROK_MODEL_NAME = "grok-3"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    assert isinstance(settings.GROK_API_KEY, SecretStr)

    _stub_load_model(monkeypatch)

    # No ctor args: everything should come from Settings
    model = GrokModel()

    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client sees the env/Settings value
    assert kw.get("api_key") == "env-secret-key"
    # Model name from Settings
    assert model.name == "grok-3"


##############################
# calculate_cost unit tests  #
##############################


def test_grok_calculate_cost_returns_correct_value(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 0.005
        settings.GROK_COST_PER_OUTPUT_TOKEN = 0.015

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    cost = model.calculate_cost(input_tokens=400, output_tokens=200)
    expected = 400 * 0.005 + 200 * 0.015
    assert cost == expected


def test_grok_calculate_cost_returns_none_when_prices_missing(
    monkeypatch, settings
):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    model.model_data.input_price = None
    model.model_data.output_price = None

    cost = model.calculate_cost(input_tokens=400, output_tokens=200)
    assert cost is None


def test_grok_calculate_cost_with_zero_tokens(monkeypatch, settings):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 0.005
        settings.GROK_COST_PER_OUTPUT_TOKEN = 0.015

    _stub_load_model(monkeypatch)

    model = GrokModel(model="grok-3")
    cost = model.calculate_cost(input_tokens=0, output_tokens=0)
    assert cost == 0.0


#####################################
# generate()/a_generate() content
#####################################


class _FakeGrokResponse(SimpleNamespace):
    def __init__(self):
        super().__init__(
            content="fake output",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        )


class _FakeGrokChat:
    """Records appended messages; sample() never touches the network."""

    def __init__(self, async_mode: bool):
        self.async_mode = async_mode
        self.messages = []

    def create(self, **kwargs):
        return self

    def append(self, message):
        self.messages.append(message)

    def sample(self):
        if self.async_mode:
            return self._a_sample()
        return _FakeGrokResponse()

    async def _a_sample(self):
        return _FakeGrokResponse()


def _stub_grok_chat(monkeypatch, async_mode: bool) -> _FakeGrokChat:
    """Stub load_model() so the recorded chat can be inspected."""
    chat = _FakeGrokChat(async_mode)

    def fake_load_model(self, async_mode: bool = False):
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr(
        grok_mod.GrokModel,
        "load_model",
        fake_load_model,
        raising=True,
    )
    return chat


def _grok_model(monkeypatch, settings, async_mode: bool):
    with settings.edit(persist=False):
        settings.GROK_API_KEY = "test-key"
        settings.GROK_COST_PER_INPUT_TOKEN = 1e-6
        settings.GROK_COST_PER_OUTPUT_TOKEN = 1e-6

    chat = _stub_grok_chat(monkeypatch, async_mode=async_mode)
    model = GrokModel(model="grok-3")
    return model, chat


def test_grok_generate_sends_native_text_content(monkeypatch, settings):
    """
    generate() must hand xai_sdk.chat.user() content it accepts (str or
    chat_pb2.Content), not OpenAI-style dicts, otherwise message construction
    raises TypeError before any request is sent.
    """
    pytest.importorskip("xai_sdk")
    from xai_sdk.proto import chat_pb2

    model, chat = _grok_model(monkeypatch, settings, async_mode=False)

    output, _ = model.generate("What is 2+2?")

    assert output == "fake output"
    [message] = chat.messages
    assert message.role == chat_pb2.MessageRole.ROLE_USER
    [block] = message.content
    assert block.text == "What is 2+2?"


async def test_grok_a_generate_sends_native_text_content(monkeypatch, settings):
    """
    a_generate() must hand xai_sdk.chat.user() content it accepts (str or
    chat_pb2.Content), not OpenAI-style dicts, otherwise message construction
    raises TypeError before any request is sent.
    """
    pytest.importorskip("xai_sdk")
    from xai_sdk.proto import chat_pb2

    model, chat = _grok_model(monkeypatch, settings, async_mode=True)

    output, _ = await model.a_generate("What is 2+2?")

    assert output == "fake output"
    [message] = chat.messages
    assert message.role == chat_pb2.MessageRole.ROLE_USER
    [block] = message.content
    assert block.text == "What is 2+2?"


def test_grok_generate_sends_native_multimodal_content(monkeypatch, settings):
    """
    Multimodal prompts must become native xai_sdk content blocks (text +
    image), not OpenAI-style dicts, otherwise user() raises TypeError.
    """
    pytest.importorskip("xai_sdk")

    model, chat = _grok_model(monkeypatch, settings, async_mode=False)

    image = MLLMImage(url="https://example.com/cat.png")
    output, _ = model.generate(f"What is in this image? {image}")

    assert output == "fake output"
    [message] = chat.messages
    first, second = message.content
    assert first.text == "What is in this image? "
    assert second.image_url.image_url == "https://example.com/cat.png"
