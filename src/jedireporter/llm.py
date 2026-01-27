import os

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from importlib.resources import open_text
from typing import Any, ClassVar, Dict, Type, TypeVar

import boto3
import httpx
import instructor
import yaml

from botocore.config import Config as BedrockConfig
from langfuse import get_client, Langfuse, observe
from langfuse.api import UnauthorizedError
from langfuse.openai import OpenAI as LangfuseOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, model_validator

from jedireporter.camelModel import CamelModel
from jedireporter.utils import dict_utils as dictutil
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)

T = TypeVar('T', bound=BaseModel)


class LLMProvider(str, Enum):
    OPENAI = 'openai'
    BEDROCK = 'bedrock'
    MISTRAL = 'mistral'


class AwsCredentialEnvConfig(BaseModel):
    access_key_id_env: str
    secret_access_key_env: str
    region_env: str
    role_env: str


class AwsCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    region: str


class LLMProfile(CamelModel):
    name: str
    model_version: str
    provider: LLMProvider
    api_key_env: str | None = None
    aws_credentials_env: AwsCredentialEnvConfig | None = None
    temperature: float | None = None
    max_out_tokens: int | None = None
    reasoning_effort: str | None = None
    thinking_tokens: int | None = None

    @property
    def instructor_name(self) -> str:
        return self.provider.value + '/' + self.model_version

    @cached_property
    def api_key(self) -> str | None:
        if self.provider is LLMProvider.BEDROCK:
            return None     # Bedrock relies on AWS credentials mentioned in `.env_template`
        if self.api_key_env is None:
            raise ValueError(
                f'Environment variable name must be defined in "api_key_env" for {self.provider} provider.'
            )
        if value := os.getenv(self.api_key_env):
            return value
        raise ValueError(f'Environment variable "{self.api_key_env}" is not set')

    @cached_property
    def aws_credentials(self) -> AwsCredentials | None:
        if self.provider in (LLMProvider.OPENAI, LLMProvider.MISTRAL):
            return None     # OpenAI and Mistral rely on API key mentioned in `.env_template`
        env_config = self.aws_credentials_env
        if env_config is None:
            raise ValueError(
                f'AWS credential environment variables must be configured for {self.provider} provider.'
            )

        def read_env(env_name: str) -> str:
            if value := os.getenv(env_name):
                return value
            raise ValueError(f'Environment variable "{env_name}" is not set')

        return AwsCredentials(
            access_key_id=read_env(env_config.access_key_id_env),
            secret_access_key=read_env(env_config.secret_access_key_env),
            region=read_env(env_config.region_env),
        )

    @cached_property
    def aws_role(self) -> str | None:
        if self.provider in (LLMProvider.OPENAI, LLMProvider.MISTRAL):
            return None
        if self.aws_credentials_env is None or self.aws_credentials_env.role_env is None:
            raise ValueError(
                f'Environment variable name must be defined in "aws_credentials_env.role_env" '
                f'for {self.provider} provider.'
            )
        if value := os.getenv(self.aws_credentials_env.role_env):
            return value
        raise ValueError(f'Environment variable "{self.aws_credentials_env.role_env}" is not set')

    @model_validator(mode='after')
    def check_provider_config(self):
        """Validate that env var names are configured for the provider, without checking their values."""
        if self.provider is LLMProvider.BEDROCK:
            if self.aws_credentials_env is None:
                raise ValueError(
                    f'AWS credential environment variable names must be configured in "aws_credentials_env" '
                    f'for {self.provider} provider.'
                )
        else:
            if self.api_key_env is None:
                raise ValueError(
                    f'Environment variable name must be defined in "api_key_env" for {self.provider} provider.'
                )
        return self


class LLMProfileLoader:
    _PROFILES: ClassVar[dict[str, LLMProfile]] = {}

    @classmethod
    def default(cls) -> LLMProfile:
        return cls.get('default')

    @classmethod
    def get(cls, name: str) -> LLMProfile:
        config = cls._get_config(name)
        return config

    @classmethod
    def _get_config(cls, name: str) -> LLMProfile:
        cls._ensure_loaded()
        try:
            return cls._PROFILES[name]
        except KeyError as exc:
            raise ValueError(f'Unknown LLM profile "{name}"') from exc

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._PROFILES:
            return

        with open_text('jedireporter.resources', 'llm-configs.yaml', encoding='utf-8') as f:
            config_obj = yaml.safe_load(f)
            model_entries = config_obj.get('models') if config_obj else None
            if not model_entries:
                raise ValueError('No models defined in llm-configs.yaml')

            for entry in model_entries:
                validated_profile = LLMProfile.model_validate(entry)
                cls._PROFILES[validated_profile.name] = validated_profile
        if 'default' not in cls._PROFILES:
            raise ValueError('Missing default LLM profile')


class InstructorLLM(ABC):
    _client: instructor.Instructor
    _langfuse_client: Langfuse
    _profile: LLMProfile
    _max_retries: int

    _PROVIDER_REGISTRY: ClassVar[Dict[LLMProvider, Type['InstructorLLM']]] = {}

    def __init_subclass__(cls, *, provider: LLMProvider | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if provider:
            if provider in InstructorLLM._PROVIDER_REGISTRY and InstructorLLM._PROVIDER_REGISTRY[provider] is not cls:
                raise ValueError(f'Provider "{provider.value}" already registered')
            InstructorLLM._PROVIDER_REGISTRY[provider] = cls

    def __init__(self, profile: LLMProfile, max_retries: int = 3) -> None:
        self._profile = profile
        self._max_retries = max_retries
        self._client = self._build_client()
        self._langfuse_client: Langfuse | None = get_client()
        try:
            self._langfuse_client.auth_check()
        except httpx.ConnectTimeout as exc:
            base_url = os.environ.get('LANGFUSE_BASE_URL')
            LOG.warning(f'Failed to connect to {base_url} Langfuse instance, with following exception: {str(exc)}. '
                        f'Logging to Langfuse is disabled.')
            self._langfuse_client = None
        except UnauthorizedError as exc:
            LOG.warning(f'Provided credentials for Langfuse access are invalid, check the env vars, with following '
                        f'exception: {str(exc)}. Logging to Langfuse is disabled.')
            self._langfuse_client = None

    @classmethod
    def from_profile(cls, profile: LLMProfile, *, max_retries: int = 3) -> 'InstructorLLM':
        try:
            impl = cls._PROVIDER_REGISTRY[profile.provider]
        except KeyError as exc:
            raise ValueError(f'No InstructorLLM registered for provider "{profile.provider.value}"') from exc
        return impl(profile=profile, max_retries=max_retries)

    @classmethod
    def default(cls, *, max_retries: int = 3) -> 'InstructorLLM':
        return cls.from_profile(LLMProfileLoader.default(), max_retries=max_retries)

    @abstractmethod
    def _build_client(self) -> instructor.Instructor:
        pass

    @staticmethod
    def _init_message(
            system_prompt: str | None = None,
            user_prompt: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        message = []
        if system_prompt:
            message.append({'role': 'system', 'content': system_prompt})
        if user_prompt:
            message.append({'role': 'user', 'content': user_prompt})
        return message

    @abstractmethod
    def get_completion(
            self,
            user_prompt: str,
            *,
            structured_output_type: Type[T],
            system_prompt: str | None = None,
            strict: bool = True,
            tools: list[str] | None = None,
    ) -> T:
        pass

    def log_to_langfuse(self, kwargs: dict[str, Any], response: T, usage: dict[str, Any]) -> None:
        """Logs LLM call with various parameters to Langfuse. Currently Anthropic models via Amazon Bedrock and Mistral
        providers are supported. Some parameters will be most likely missing in the Langfuse for other providers, due
        to different parameter naming."""
        # Skip logging if Langfuse is not configured
        if self._langfuse_client is None:
            return
        # Informs the user, that for different providers, we cannot ensure all parameters will be logged to langfuse
        if self._profile.provider not in (LLMProvider.BEDROCK, LLMProvider.MISTRAL):
            LOG.warning(f'Langfuse wrapper is implemented for {LLMProvider.BEDROCK} and {LLMProvider.MISTRAL} providers'
                        f', it is probable that most parameters for the {self._profile.provider} provider will not be'
                        f' correctly logged to Langfuse, due to the naming nuances across providers.')

        temperature = kwargs.get('temperature') if kwargs.get('temperature') is not None else (
            dictutil.getValue(kwargs, 'inferenceConfig.temperature'))
        model_params = {
            'max_tokens': kwargs.get('max_tokens') or dictutil.getValue(kwargs, 'inferenceConfig.maxTokens'),
            'temperature': temperature,
            'reasoning_tokens': dictutil.getValue(kwargs, 'additionalModelRequestFields.thinking.budget_tokens')
        }
        metadata = {
            'strict_output_validation': kwargs.get('strict'),
            'max_retries': kwargs.get('max_retries'),
            'provider': self._profile.provider
        }
        usage_details = {'input': usage.get('inputTokens') or usage.get('prompt_tokens'),
                         'output': usage.get('outputTokens') or usage.get('completion_tokens'),
                         'total': usage.get('totalTokens') or usage.get('total_tokens')}
        self._langfuse_client.update_current_generation(
            input=kwargs.get('messages'),
            model=self._profile.model_version,
            model_parameters=model_params,
            metadata=metadata,
            output=response,
            usage_details=usage_details
        )


class OpenAIInstructorLLM(InstructorLLM, provider=LLMProvider.OPENAI):

    def _build_client(self) -> instructor.Instructor:
        client = LangfuseOpenAI(api_key=self._profile.api_key)
        return instructor.from_openai(client, mode=instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS)

    def get_completion(
            self,
            user_prompt: str,
            *,
            structured_output_type: Type[T],
            system_prompt: str | None = None,
            strict: bool = True,
            tools: list[str] | None = None,
    ) -> T:
        kwargs: dict[str, Any] = {
            'input': user_prompt,
            'model': self._profile.model_version,
        }
        if self._profile.max_out_tokens is not None and self._profile.max_out_tokens > 0:
            kwargs['max_output_tokens'] = self._profile.max_out_tokens
        if system_prompt:
            kwargs['instructions'] = system_prompt
        if self._profile.temperature is not None:
            kwargs['temperature'] = self._profile.temperature
        elif self._profile.reasoning_effort:
            kwargs['reasoning'] = {'effort': self._profile.reasoning_effort}
        if tools:
            kwargs['tools'] = [{'type': tool} for tool in tools]

        response, raw_response = self._client.responses.create_with_completion(
            response_model=structured_output_type,
            max_retries=self._max_retries,
            strict=strict,
            **kwargs,
        )
        assert isinstance(response, structured_output_type), (
            f'Expected {structured_output_type}, but got: {type(response)}'
        )
        LOG.debug(f'Full raw response from LLM: {raw_response.model_dump_json()}')
        return response


class BedrockInstructorLLM(InstructorLLM, provider=LLMProvider.BEDROCK):

    def _build_client(self) -> instructor.Instructor:
        if self._profile.aws_role is None:
            raise ValueError('Missing AWS IAM role')
        sts = boto3.client('sts',
                           region_name=self._profile.aws_credentials.region,
                           aws_access_key_id=self._profile.aws_credentials.access_key_id,
                           aws_secret_access_key=self._profile.aws_credentials.secret_access_key)
        response = sts.assume_role(RoleArn=self._profile.aws_role, RoleSessionName='atex-jedi-session')
        credentials = response['Credentials']
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self._profile.aws_credentials.region,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            config=BedrockConfig(read_timeout=1200),    # Prevent read timeouts for long LLM tasks (20 mins)
        )
        return instructor.from_bedrock(bedrock_client)

    @observe(as_type='generation', name='bedrock-instructor')
    def get_completion(
            self,
            user_prompt: str,
            *,
            structured_output_type: Type[T],
            system_prompt: str | None = None,
            strict: bool = True,
            tools: list[str] | None = None,
    ) -> T:
        if tools:
            raise NotImplementedError('Tool calling is not supported for Bedrock provider in this wrapper.')
        messages = self._init_message(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        kwargs: dict[str, Any] = {
            'messages': messages,
            'response_model': structured_output_type,
            'max_retries': self._max_retries,
            'modelId': self._profile.model_version,
            'strict': strict,
        }

        inference_config: dict[str, Any] = {}
        if self._profile.max_out_tokens is not None and self._profile.max_out_tokens > 0:
            inference_config['maxTokens'] = self._profile.max_out_tokens
        if self._profile.temperature is not None:
            inference_config['temperature'] = self._profile.temperature

        if inference_config:
            kwargs['inferenceConfig'] = inference_config

        if self._profile.thinking_tokens:
            if self._profile.thinking_tokens >= self._profile.max_out_tokens:
                raise ValueError(f'Limit of thinking tokens {self._profile.thinking_tokens} must be less than the limit'
                                 f' of maximum output tokens {self._profile.max_out_tokens}')
            else:
                kwargs['additionalModelRequestFields'] = {}
                kwargs['additionalModelRequestFields']['thinking'] = {
                    'type': 'enabled', 'budget_tokens': self._profile.thinking_tokens}
                if 'temperature' in inference_config:
                    raise ValueError('Setting both temperature and thinking tokens is not supported by the Anthropic.')

        response, raw_response = self._client.create_with_completion(**kwargs)
        LOG.debug(f'Full raw response from LLM: {raw_response}')

        assert isinstance(response, structured_output_type), (
            f'Expected {structured_output_type}, but got: {type(response)}'
        )

        # Langfuse log
        try:
            self.log_to_langfuse(kwargs=kwargs, response=response, usage=raw_response.get('usage', {}))
        except Exception as exc:
            LOG.exception(f'Failed to update Langfuse generation (post-call) for Bedrock with following exception: '
                          f'{str(exc)}.')
        return response


class MistralInstructorLLM(InstructorLLM, provider=LLMProvider.MISTRAL):

    def _build_client(self) -> instructor.Instructor:
        client = instructor.from_provider(
            f'mistral/{self._profile.model_version}',
            mode=instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS,
            api_key=self._profile.api_key
        )
        return client

    @observe(as_type='generation', name='mistral-instructor')
    def get_completion(
            self,
            user_prompt: str,
            *,
            structured_output_type: Type[T],
            system_prompt: str | None = None,
            strict: bool = True,
            tools: list[str] | None = None,
    ) -> T:
        messages = self._init_message(system_prompt=system_prompt, user_prompt=user_prompt)

        kwargs: dict[str, Any] = {
            'messages': messages,
            'response_model': structured_output_type,
            'max_retries': self._max_retries,
            'strict': strict,
            'parallel_tool_calls': False
        }
        if self._profile.max_out_tokens is not None:
            kwargs['max_tokens'] = self._profile.max_out_tokens
        if self._profile.temperature is not None:
            kwargs['temperature'] = self._profile.temperature
        if tools:
            kwargs['tools'] = [{'type': tool} for tool in tools]

        response, raw_response = self._client.chat.completions.create_with_completion(**kwargs)
        LOG.debug(f'Full raw response from LLM: {raw_response}')
        assert isinstance(response, structured_output_type), (
            f'Expected {structured_output_type}, but got: {type(response)}'
        )

        # Langfuse log
        try:
            self.log_to_langfuse(kwargs=kwargs, response=response, usage=raw_response.usage.model_dump())
        except Exception as exc:
            LOG.exception(f'Failed to update Langfuse generation (post-call) for Mistral with following exception: '
                          f'{str(exc)}.')
        return response
