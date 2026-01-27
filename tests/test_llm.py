import logging
from typing import Any
from unittest.mock import Mock

import pytest
from instructor import Instructor

from jedireporter.llm import (
    AwsCredentialEnvConfig, BedrockInstructorLLM, InstructorLLM, LLMProfile, LLMProfileLoader,
    LLMProvider, MistralInstructorLLM, OpenAIInstructorLLM,
)


@pytest.fixture(autouse=True)
def setup_envs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('JEDI_OPENAI_API_KEY', 'api-key')
    monkeypatch.setenv('JEDI_MISTRAL_API_KEY', 'api-key')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'aws-access-key')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'aws-secret-key')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'eu-west-1')
    monkeypatch.setenv('AWS_BEDROCK_ROLE', 'arn:aws:iam::123456789012:role/TestBedrockRole')


@pytest.fixture
def default_profile() -> LLMProfile:
    return LLMProfile(
        name='default',
        model_version='gpt-5.1-2025-11-13',
        provider=LLMProvider.OPENAI,
        api_key_env='JEDI_OPENAI_API_KEY',
        max_out_tokens=64000,
        reasoning_effort='medium'
    )


class TestLLMProfileLoader:
    @pytest.fixture
    def gpt_5_profile(self) -> LLMProfile:
        return LLMProfile(
            name='gpt-5',
            model_version='gpt-5-2025-08-07',
            provider=LLMProvider.OPENAI,
            api_key_env='JEDI_OPENAI_API_KEY',
            max_out_tokens=32768,
            reasoning_effort='medium'
        )

    @pytest.fixture
    def gpt_4_1_profile(self) -> LLMProfile:
        return LLMProfile(
            name='gpt-4.1',
            model_version='gpt-4.1-2025-04-14',
            provider=LLMProvider.OPENAI,
            api_key_env='JEDI_OPENAI_API_KEY',
            max_out_tokens=32768,
            temperature=0.0
        )

    @pytest.fixture
    def gpt_5_2_profile(self) -> LLMProfile:
        return LLMProfile(
            name='gpt-5.2',
            model_version='gpt-5.2-2025-12-11',
            provider=LLMProvider.OPENAI,
            api_key_env='JEDI_OPENAI_API_KEY',
            max_out_tokens=64000,
            reasoning_effort='medium'
        )

    @pytest.fixture
    def aws_env_credentials_config(self) -> AwsCredentialEnvConfig:
        return AwsCredentialEnvConfig(
            access_key_id_env='AWS_ACCESS_KEY_ID',
            secret_access_key_env='AWS_SECRET_ACCESS_KEY',
            region_env='AWS_DEFAULT_REGION',
            role_env='AWS_BEDROCK_ROLE',
        )

    @pytest.fixture
    def claude_profile(self, aws_env_credentials_config: AwsCredentialEnvConfig) -> LLMProfile:
        return LLMProfile(
            name='claude',
            model_version='eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
            provider=LLMProvider.BEDROCK,
            thinking_tokens=4096,
            max_out_tokens=32768,
            aws_credentials_env=aws_env_credentials_config,
        )

    @pytest.fixture
    def mistral_profile(self) -> LLMProfile:
        return LLMProfile(
            name='mistral',
            model_version='mistral-large-2512',
            provider=LLMProvider.MISTRAL,
            api_key_env='JEDI_MISTRAL_API_KEY',
            max_out_tokens=32768,
            temperature=0.0
        )

    def test_missing_config_raises_on_creation(self) -> None:
        """Missing env var names should raise during profile creation."""
        with pytest.raises(ValueError, match='api_key_env'):
            LLMProfile(
                name='test-openai',
                model_version='gpt-test',
                provider=LLMProvider.OPENAI,
                # api_key_env is missing
            )
        with pytest.raises(ValueError, match='aws_credentials_env'):
            LLMProfile(
                name='test-bedrock',
                model_version='claude-test',
                provider=LLMProvider.BEDROCK,
                # aws_credentials_env is missing
            )

    def test_missing_credentials_raises_on_access(self) -> None:
        """Missing env var values should raise when credentials are accessed, not during creation."""
        # Profile creation should succeed even with unset env vars
        openai_profile = LLMProfile(
            name='test-openai',
            model_version='gpt-test',
            provider=LLMProvider.OPENAI,
            api_key_env='UNSET_KEY',
        )
        bedrock_profile = LLMProfile(
            name='test-bedrock',
            model_version='claude-test',
            provider=LLMProvider.BEDROCK,
            aws_credentials_env=AwsCredentialEnvConfig(
                access_key_id_env='UNSET',
                secret_access_key_env='UNSET',
                region_env='UNSET',
                role_env='UNSET_ROLE',
            ),
        )
        # But accessing credentials should fail
        with pytest.raises(ValueError, match='UNSET_KEY'):
            _ = openai_profile.api_key
        with pytest.raises(ValueError, match='UNSET'):
            _ = bedrock_profile.aws_credentials

    def test_default_profile(self, default_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('default')
        assert actual == default_profile

    def test_gpt_4_1_profile(self, gpt_4_1_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('gpt-4.1')
        assert actual == gpt_4_1_profile

    def test_gpt_5_profile(self, gpt_5_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('gpt-5')
        assert actual == gpt_5_profile

    def test_gpt_5_2_profile(self, gpt_5_2_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('gpt-5.2')
        assert actual == gpt_5_2_profile

    def test_claude_profile(self, claude_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('claude')
        assert actual == claude_profile

    def test_mistral_profile(self, mistral_profile: LLMProfile) -> None:
        actual = LLMProfileLoader.get('mistral')
        assert actual == mistral_profile

    def test_unknown_profile(self) -> None:
        with pytest.raises(ValueError):
            LLMProfileLoader.get('unknown')


class TestInstructorLLM:
    def test_from_profile_instantiates_registered_provider(self, default_profile: LLMProfile) -> None:
        OpenAIInstructorLLM._build_client = Mock()
        instance = InstructorLLM.from_profile(default_profile)

        assert isinstance(instance, OpenAIInstructorLLM)

    def test_duplicate_provider_registration(self) -> None:
        with pytest.raises(ValueError):
            class _DuplicateOpenAI(InstructorLLM, provider=LLMProvider.OPENAI):

                def _build_client(self) -> Instructor:
                    raise AssertionError('should not be called')

                def get_completion(self, *args: Any, **kwargs: Any) -> None:
                    raise AssertionError('should not be called')

    def test_provider_registry_contains_all_subclasses(self) -> None:
        registry = InstructorLLM._PROVIDER_REGISTRY

        assert registry[LLMProvider.OPENAI] is OpenAIInstructorLLM
        assert registry[LLMProvider.BEDROCK] is BedrockInstructorLLM

    def test_default_factory_uses_profile(self) -> None:
        OpenAIInstructorLLM._build_client = Mock()

        instance = InstructorLLM.default(max_retries=7)

        assert isinstance(instance, OpenAIInstructorLLM)
        assert instance._max_retries == 7

    @pytest.mark.parametrize(
        ('model', 'kwargs', 'usage', 'expected_model_params', 'expected_usage'),
        [
            (
                'mistral',
                {
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                    'response_model': dict,
                    'max_retries': 2,
                    'strict': False,
                    'max_tokens': 50,
                    'temperature': 0.2,
                },
                {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                {'max_tokens': 50, 'temperature': 0.2, 'reasoning_tokens': None},
                {'input': 10, 'output': 5, 'total': 15},
            ),
            (
                'claude',
                {
                    'messages': [{'role': 'user', 'content': 'Hi again'}],
                    'response_model': dict,
                    'max_retries': 3,
                    'modelId': 'bedrock-model',
                    'strict': True,
                    'inferenceConfig': {'maxTokens': 64},
                    'additionalModelRequestFields': {'thinking': {'type': 'enabled', 'budget_tokens': 200}},
                },
                {'inputTokens': 6, 'outputTokens': 2, 'totalTokens': 8},
                {'max_tokens': 64, 'temperature': None, 'reasoning_tokens': 200},
                {'input': 6, 'output': 2, 'total': 8},
            ),
        ],
    )
    def test_log_to_langfuse(
            self,
            model: str,
            kwargs: dict[str, Any],
            usage: dict[str, Any],
            expected_model_params: dict[str, Any],
            expected_usage: dict[str, Any],
    ) -> None:
        profile = LLMProfileLoader.get(model)
        llm_cls = {
            LLMProvider.MISTRAL: MistralInstructorLLM,
            LLMProvider.BEDROCK: BedrockInstructorLLM,
        }[profile.provider]
        llm = object.__new__(llm_cls)
        llm._profile = profile
        llm._langfuse_client = Mock()

        llm.log_to_langfuse(kwargs=kwargs, response={'data': 'ok'}, usage=usage)

        llm._langfuse_client.update_current_generation.assert_called_once_with(
            input=kwargs.get('messages'),
            model=profile.model_version,
            model_parameters=expected_model_params,
            metadata={
                'strict_output_validation': kwargs.get('strict'),
                'max_retries': kwargs.get('max_retries'),
                'provider': profile.provider
            },
            output={'data': 'ok'},
            usage_details=expected_usage
        )

    def test_log_to_langfuse_warns_for_openai(self, default_profile: LLMProfile,
                                              caplog: pytest.LogCaptureFixture) -> None:
        llm = object.__new__(OpenAIInstructorLLM)
        llm._profile = default_profile
        llm._langfuse_client = Mock()

        kwargs = {
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'temperature': 0.3,
            'max_tokens': 321,
            'max_retries': 4,
            'strict': True,
        }
        usage = {'inputTokens': 7, 'outputTokens': 3, 'totalTokens': 10}
        caplog.set_level(logging.WARNING, logger='jedireporter.llm')

        llm.log_to_langfuse(kwargs=kwargs, response=Mock(), usage=usage)
        llm._langfuse_client.update_current_generation.assert_called_once()
        assert any('Langfuse wrapper is implemented for' in record.message for record in caplog.records)
