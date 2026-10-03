"""Unit tests for OpenAI-compatible STT & LLM universal adapters and settings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import get_masked_settings, mask_secret, save_config, settings
from llm import llm_registry
from llm.openai_compat import OpenAICompatibleLLMProvider
from main import app
from transcribers.base import Segment
from transcribers.factory import transcriber_factory
from transcribers.openai_compat import OpenAICompatibleTranscriber

client = TestClient(app)


def test_mask_secret():
    assert mask_secret(None) == ""
    assert mask_secret("") == ""
    assert mask_secret("short") == "••••••••"
    assert mask_secret("gsk_1234567890abcdef") == "gsk_••••cdef"
    assert mask_secret("sk-or-v1-abcdef123456789") == "sk-o••••6789"


def test_openai_transcriber_initialization():
    fake_key = "gsk_test123456789"
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1/",
        api_key_getter=lambda: fake_key,
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo", "whisper-large-v3"],
        extra_headers={"X-Custom": "TestVal"},
    )

    assert transcriber.name == "groq"
    assert transcriber.display_name == "Groq Cloud Whisper"
    assert transcriber.get_base_url() == "https://api.groq.com/openai/v1"
    assert transcriber.get_api_key() == fake_key
    assert transcriber.is_available() is True
    assert transcriber.get_extra_headers() == {"X-Custom": "TestVal"}


def test_openai_transcriber_unconfigured_availability():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter=lambda: None,
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )
    assert transcriber.is_available() is False
    with pytest.raises(RuntimeError, match="API key is not configured"):
        transcriber.transcribe("dummy_file.wav")


def test_openai_transcriber_parse_verbose_json():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="fake_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    verbose_response = {
        "text": "Hello world and welcome to speech recognition.",
        "language": "english",
        "duration": 5.2,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Hello world",
                "avg_logprob": -0.105,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.99},
                    {"word": "world", "start": 1.1, "end": 2.5, "probability": 0.98},
                ],
            },
            {
                "id": 1,
                "start": 2.6,
                "end": 5.2,
                "text": " and welcome to speech recognition.",
                "avg_logprob": -0.05,
                "words": [
                    {"word": "and", "start": 2.6, "end": 3.0, "probability": 0.95},
                    {"word": "welcome", "start": 3.1, "end": 5.2, "probability": 0.97},
                ],
            },
        ],
    }

    emitted_segments: list[Segment] = []
    segs, text, lang, dur = transcriber._parse_verbose_json(
        data=verbose_response,
        time_offset=10.0,
        start_segment_id=5,
        on_segment=lambda s: emitted_segments.append(s),
    )

    assert len(segs) == 2
    assert len(emitted_segments) == 2
    assert segs[0].id == 5
    assert segs[0].start == 10.0
    assert segs[0].end == 12.5
    assert segs[0].text == "Hello world"
    assert segs[0].confidence is not None
    assert len(segs[0].words) == 2
    assert segs[0].words[0].word == "Hello"
    assert segs[0].words[0].start == 10.0
    assert segs[0].words[0].end == 11.0

    assert segs[1].id == 6
    assert segs[1].start == 12.6
    assert segs[1].end == 15.2
    assert lang == "english"
    assert dur == 5.2


def test_openai_transcriber_transcribe_with_mock(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="gsk_valid_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Transcribed test content.",
        "language": "en",
        "duration": 3.0,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 3.0,
                "text": "Transcribed test content.",
                "words": [
                    {"word": "Transcribed", "start": 0.0, "end": 1.0},
                    {"word": "test", "start": 1.1, "end": 1.8},
                    {"word": "content.", "start": 1.9, "end": 3.0},
                ],
            }
        ],
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        result = transcriber.transcribe(audio_file, model_name="whisper-large-v3-turbo")

        assert result.text == "Transcribed test content."
        assert len(result.segments) == 1
        assert len(result.segments[0].words) == 3
        assert result.duration == 3.0
        assert result.language == "en"

        # Check call arguments
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.groq.com/openai/v1/audio/transcriptions"
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers["Authorization"] == "Bearer gsk_valid_key"


def test_openai_transcriber_numpy_input():
    transcriber = OpenAICompatibleTranscriber(
        name="groq",
        display_name="Groq Cloud Whisper",
        base_url_getter="https://api.groq.com/openai/v1",
        api_key_getter="gsk_valid_key",
        default_model="whisper-large-v3-turbo",
        supported_models=["whisper-large-v3-turbo"],
    )

    dummy_pcm = np.zeros(16000, dtype=np.float32)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "NumPy test passed.",
        "language": "en",
        "duration": 1.0,
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "NumPy test passed.", "words": []}],
    }

    with patch("httpx.Client.post", return_value=mock_response):
        result = transcriber.transcribe(dummy_pcm)
        assert result.text == "NumPy test passed."


def test_openai_transcriber_fallback_when_granularities_unsupported(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    transcriber = OpenAICompatibleTranscriber(
        name="custom",
        display_name="Custom STT",
        base_url_getter="https://api.custom.ai/v1",
        api_key_getter="sk_custom",
        default_model="whisper-1",
        supported_models=["whisper-1"],
    )

    err_response = MagicMock()
    err_response.status_code = 400
    err_response.text = "Unrecognized parameter: timestamp_granularities"

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {
        "text": "Fallback succeeded.",
        "language": "en",
        "duration": 2.0,
        "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "Fallback succeeded.", "words": []}],
    }

    with patch("httpx.Client.post", side_effect=[err_response, ok_response]) as mock_post:
        result = transcriber.transcribe(audio_file)
        assert result.text == "Fallback succeeded."
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_openai_llm_provider_streaming_mock():
    provider = OpenAICompatibleLLMProvider(
        provider_id="groq",
        display_name="Groq Cloud",
        api_key_getter="gsk_test",
        base_url_getter="https://api.groq.com/openai/v1",
        default_model_getter="llama-3.3-70b-versatile",
    )

    assert provider.is_configured() is True
    assert provider.get_default_model() == "llama-3.3-70b-versatile"

    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line.decode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp
        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()):
        collected = []
        async for token in provider.generate_stream("Say hello"):
            collected.append(token)
        assert "".join(collected) == "Hello world"


@pytest.mark.asyncio
async def test_openai_llm_provider_list_models():
    provider = OpenAICompatibleLLMProvider(
        provider_id="openrouter",
        display_name="OpenRouter",
        api_key_getter="sk-or-test",
        base_url_getter="https://openrouter.ai/api/v1",
        default_model_getter="meta-llama/llama-3.3-70b-instruct",
        curated_models=["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-r1"],
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "deepseek/deepseek-r1"},
            {"id": "meta-llama/llama-3.3-70b-instruct"},
            {"id": "mistralai/mistral-large"},
        ]
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        models = await provider.list_models()
        assert models[0] == "meta-llama/llama-3.3-70b-instruct"
        assert models[1] == "deepseek/deepseek-r1"
        assert "mistralai/mistral-large" in models


def test_transcriber_factory_engines_and_models():
    engines = transcriber_factory.list_engines()
    engine_ids = [e["id"] for e in engines]
    assert "groq" in engine_ids
    assert "openrouter" in engine_ids
    assert "openai" in engine_ids

    groq_models = transcriber_factory.get_models_for_engine("groq")
    assert "whisper-large-v3-turbo" in groq_models
    assert "whisper-large-v3" in groq_models

    or_models = transcriber_factory.get_models_for_engine("openrouter")
    assert "openai/whisper-large-v3" in or_models


def test_api_whisper_models_endpoint():
    resp = client.get("/api/whisper/models?engine=groq")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine"] == "groq"
    assert "whisper-large-v3-turbo" in data["models"]


def test_settings_endpoints():
    # 1. GET /api/settings
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    settings_data = resp.json()
    assert "groq" in settings_data
    assert "openrouter" in settings_data

    # 2. POST /api/settings
    update_payload = {
        "groq": {"default_model": "llama-3.3-70b-versatile"},
        "openrouter": {"default_model": "meta-llama/llama-3.3-70b-instruct"},
    }
    resp_post = client.post("/api/settings", json=update_payload)
    assert resp_post.status_code == 200
    assert resp_post.json()["status"] == "ok"


def test_settings_test_endpoint():
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        resp = client.post(
            "/api/settings/test",
            json={"provider": "groq", "api_key": "gsk_dummy_test_key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["online"] is True
        assert "latency_ms" in data
