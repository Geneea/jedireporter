import argparse
import json
import os
import sys
import traceback
from importlib.metadata import distribution
from typing import Any, Callable, Sequence

import dotenv
import pydantic
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import RedirectResponse

from jedireporter.article import Article
from jedireporter.llm import InstructorLLM, LLMProfileLoader
from jedireporter.restapi.models import APIArticle, APITranscript
from jedireporter.utils import logging
from jedireporter.workflow import _get_langfuse_callbacks, InterviewProcessor, State

LOG = logging.getLogger(__package__, __file__)
_API_VERSION_1 = 'v1'
_VER = distribution('jedireporter').version


def _create_exception_handler(status_code: int = 500, log_exception: bool = False):
    """
    Returns a JSON message response for all unhandled errors from request handlers. The response JSON body
    will show exception message and traceback (if the app runs in the debug mode).

    :param status_code: the HTTP status code to return; if not specified 500 (Server Error) status code is used
    """
    async def _exception_handler(request: Request, exc):
        exc_str = f'{type(exc).__name__}: {exc}'
        if log_exception:
            LOG.exception(f'Unhandled error: {exc_str}')

        if request.app.debug:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exc_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            return JSONResponse({'message': exc_str, 'exception': exc_text}, status_code)

        else:
            return JSONResponse({'message': exc_str}, status_code)

    return _exception_handler


async def _http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"message": exc.detail}, status_code=exc.status_code)


_bad_request_handler = _create_exception_handler(status_code=400)
_exception_handlers = {
    HTTPException: _http_exception_handler,
    json.JSONDecodeError: _bad_request_handler,
    requests.JSONDecodeError: _bad_request_handler,
    pydantic.ValidationError: _bad_request_handler,
    ValueError: _bad_request_handler,
    Exception: _create_exception_handler(status_code=500, log_exception=True),
}


class _ApiRoute(APIRoute):
    def __init__(
            self, path: str, endpoint: Callable[..., Any],
            methods: list[str] | None = None, tags: list[str] | None = None,
            include_in_schema: bool = True,
            dependencies: Sequence[Depends] | None = None,
    ) -> None:
        super().__init__(
            path, endpoint, methods=methods, tags=tags, include_in_schema=include_in_schema,
            response_model_exclude_none=True, dependencies=dependencies
        )


class Message(BaseModel):
    message: str


async def redirect_to_docs(rq: Request) -> RedirectResponse:
    root_path = rq.scope.get('root_path', '') or ''
    return RedirectResponse(url=f'{root_path}/docs', status_code=307)


async def get_status() -> Message:
    """Returns basic information about the service."""
    return Message(message=f'JEDI Reporter Service {_VER} - Ready.')


async def generate(api_transcript: APITranscript, rq: Request) -> APIArticle:
    """
    Generates an article from a transcript.

    Takes a transcript with speakers and segments, processes it through the interview
    processing pipeline, and returns a structured article with paragraphs, questions,
    and answers along with the source transcript for traceability.
    """
    # Convert API input to internal model
    transcript = api_transcript.to_internal()

    # Create the processing chain
    llm = InstructorLLM.from_profile(rq.app.state.llm_profile)
    interview_processor = InterviewProcessor(llm)
    chain = interview_processor.create_workflow()

    # Process the transcript
    input_state = State(source=transcript)
    callbacks = _get_langfuse_callbacks()
    result = await chain.ainvoke(input_state, config={'callbacks': callbacks})

    article = result.get('styled') if result else None
    assert isinstance(article, Article), f'Expected Article, got: {type(article)}'

    # Set transcript for traceability before converting to API model
    article.transcript = transcript

    # Convert internal output to API model
    return APIArticle.from_internal(article)


def create_app(
    *,
    server_path: str = '',
    llm_profile: str = '',
    debug: bool = False,
) -> FastAPI:
    dotenv.load_dotenv()

    path_prefix = f'/{_API_VERSION_1}'
    app = FastAPI(
        title='JEDI Reporter Service',
        docs_url='/docs',
        routes=[
            _ApiRoute('/', methods=['GET'], endpoint=redirect_to_docs, include_in_schema=False),
            _ApiRoute('/status', methods=['GET'], endpoint=get_status),
            _ApiRoute(f'{path_prefix}/generate-article', methods=['POST'], endpoint=generate),
        ],
        root_path=server_path or os.getenv('JEDI_SERVER_PATH', ''),
        exception_handlers=_exception_handlers,
        middleware=[
            Middleware(GZipMiddleware, minimum_size=512, compresslevel=9),
        ],
        debug=debug or os.getenv('DEBUG', 'false').lower() == 'true',
        version=_VER,
    )
    app.state.llm_profile = LLMProfileLoader.get(llm_profile or os.getenv('JEDI_LLM_PROFILE', '') or 'default')
    return app


# We need this "static" app instance for multi-worker Uvicorn setup.
# Uvicorn does not support factory functions as import strings,
# which is different from gunicorn.
APP = create_app()


def main():
    parser = argparse.ArgumentParser(description='Starts JEDI Reporter Service.')
    parser.add_argument('--bind', default='localhost',
                        help='Name or IP address of the network interface where the sever will listen.')
    parser.add_argument('--port', type=int, default=5000, help='The port to listen at.')
    logging.addLogArguments(parser)

    args = parser.parse_args()
    logging.configureFromArgs(args)

    app_import_str = __package__ + '.server:APP'
    config = uvicorn.Config(
        app=app_import_str,
        host=args.bind,
        port=args.port,
        log_config=args.logConfigFile,
        timeout_graceful_shutdown=0,
        lifespan='on',
        workers=2,
    )
    server = uvicorn.Server(config)
    LOG.info(
        f'Starting JEDI Reporter Service on {args.bind}:{args.port}, '
        f'log config "{args.logConfigFile or ""}"'
    )

    server.run()


if __name__ == '__main__':
    main()
