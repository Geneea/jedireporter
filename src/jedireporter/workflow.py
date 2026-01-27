import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from importlib.resources import read_text

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from jedireporter.article import Article
from jedireporter.create_article import CreateArticleNodes, CreateArticleState
from jedireporter.fix_transcript import FixedState, FixTranscriptNodes
from jedireporter.llm import InstructorLLM, langfuse_configured, LLMProfileLoader
from jedireporter.post_process_article import PostProcessArticleNodes, PostProcessArticleState
from jedireporter.transcript import Transcript
from jedireporter.utils import logging as logutil
from jedireporter.utils import parallel
from jedireporter.utils.cli_utils import addInOutArgGroups, argOpenIn, argOpenOut

LOG = logutil.getLogger(__package__, __file__)


def _get_langfuse_callbacks() -> list[BaseCallbackHandler]:
    """Return Langfuse callback handler if configured, otherwise empty list."""
    if langfuse_configured():
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    return []


class State(BaseModel):
    source: Transcript  # the unmodified input
    fixed: Transcript | None = None  # fixed transcript produced by the pipeline
    article: Article | None = None  # interview rewritten as an article
    styled: Article | None = None  # interview/article with additional style applied


class InterviewProcessor:
    def __init__(self, llm: InstructorLLM) -> None:
        self.llm = llm
        self._fix_transcript_nodes = FixTranscriptNodes(self.llm)
        self._fix_transcript_subgraph: CompiledStateGraph | None = None
        self._create_article_nodes = CreateArticleNodes(self.llm)
        self._create_article_subgraph: CompiledStateGraph | None = None
        self._post_process_article_nodes = PostProcessArticleNodes(self.llm)
        self._post_process_article_subgraph: CompiledStateGraph | None = None

        self.workflow = StateGraph(State)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompt = read_text(
            'jedireporter.resources.templates',
            name + '.txt'
        )
        return prompt

    def run_fix_transcript_subgraph(self, state: State) -> dict[str, Transcript]:
        if self._fix_transcript_subgraph is None:
            raise ValueError('Fixed transcript subgraph must be built before running it.')
        LOG.info(f'[{state.source.id}] Fixing transcript ...')
        callbacks = _get_langfuse_callbacks()
        response = self._fix_transcript_subgraph.invoke(FixedState(source=state.source),
                                                        config={'callbacks': callbacks})
        return {'fixed': response['fixed']}

    def run_create_article_subgraph(self, state: State) -> dict[str, Article]:
        if self._create_article_subgraph is None:
            raise ValueError('Create article subgraph must be built before running it.')
        LOG.info(f'[create_article_subgraph] source.id={state.source.id}')
        if state.fixed is None:
            raise ValueError('Fixed transcript not available before create_article step')
        # Invoke the subgraph with its own pydantic state and map the result back
        callbacks = _get_langfuse_callbacks()
        response = self._create_article_subgraph.invoke(CreateArticleState(fixed=state.fixed),
                                                        config={'callbacks': callbacks})
        return {'article': response['article']}

    def modify_style(self, state: State) -> dict[str, Article]:
        if self._post_process_article_subgraph is None:
            raise ValueError('Post-process article subgraph must be built before running it.')
        LOG.info(f'[post_process_article_subgraph] source.id={state.source.id}')
        if state.article is None:
            raise ValueError('Article not available before post_process_article step')
        # Invoke the subgraph with its own pydantic state and map the result back
        callbacks = _get_langfuse_callbacks()
        response = self._post_process_article_subgraph.invoke(
            PostProcessArticleState(article=state.article, transcript=state.fixed), config={'callbacks': callbacks})
        return {'styled': response['enriched_article']}

    def create_workflow(self) -> CompiledStateGraph:
        # Build subgraphs
        self._fix_transcript_subgraph = self._fix_transcript_nodes.build_subgraph()
        self._create_article_subgraph = self._create_article_nodes.build_subgraph()
        self._post_process_article_subgraph = self._post_process_article_nodes.build_subgraph()

        self.workflow.add_sequence(
            [self.run_fix_transcript_subgraph, self.run_create_article_subgraph, self.modify_style])
        self.workflow.add_edge(START, 'run_fix_transcript_subgraph')
        self.workflow.add_edge('modify_style', END)

        chain = self.workflow.compile()
        return chain


def main():
    _version = version("jedireporter")
    argparser = argparse.ArgumentParser(
        prog='jedireporter',
        description='Process transcript JSON through the workflow. Accepts a single transcript or batch of transcripts'
                    ' but always in JSON-per-line format.'
    )
    argparser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {_version}',
    )
    addInOutArgGroups(argparser, in_title='Path to the transcript', out_title='Output article file')
    argparser.add_argument(
        '--include-transcript',
        action='store_true',
        help='Populate the transcript field in the final Article output'
    )
    argparser.add_argument(
        '--max-concurrency',
        type=int,
        default=4,
        help='Maximum number of parallel workflow invocations when a batch is provided'
    )
    argparser.add_argument(
        '--llm-profile',
        default='default',
        help='Name of the LLM profile to load from llm-configs.yaml'
    )

    logutil.addLogArguments(argparser)
    args = argparser.parse_args()
    logutil.configureFromArgs(args)

    load_dotenv()

    llm_profile = LLMProfileLoader.get(args.llm_profile)
    llm = InstructorLLM.from_profile(llm_profile)
    interview_processor = InterviewProcessor(llm)
    chain = interview_processor.create_workflow()

    def _invoke(state: State) -> dict[str, Transcript | Article] | None:
        try:
            callbacks = _get_langfuse_callbacks()
            return chain.invoke(state, config={'callbacks': callbacks})
        except Exception:
            LOG.exception(f'Failed to process transcript {state.source.id}')
            return None

    with argOpenIn(args) as finp, argOpenOut(args) as fout:
        states = (State(source=Transcript.model_validate_json(transcript)) for transcript in finp)
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            results = parallel.parallelMap(
                executor,
                _invoke,
                states,
                threadCount=args.max_concurrency,
            )

            LOG.info('Started')

            for result in filter(None, results):
                article = result.get('styled')
                if args.include_transcript:
                    # Currently it is necessary to include the transcript for the correct translation
                    # "speaker_id" -> speaker name if available in the final markdown
                    article.transcript = result.get('fixed').model_copy(deep=True)
                print(article.model_dump_json(), file=fout)

            LOG.info('Finished')


if __name__ == '__main__':
    main()
