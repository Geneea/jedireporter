from importlib.resources import read_text

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from jedireporter.article import Article, ArticleMetadata, Paragraph, WebSearchResultList
from jedireporter.llm import InstructorLLM
from jedireporter.transcript import Timecodes, Transcript
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)


class PostProcessArticleState(BaseModel):
    """
    Pydantic state for the post-process subgraph.
    Contains the produced article and accumulates enrichments from external sources.
    """
    article: Article
    transcript: Transcript
    web_search_data: WebSearchResultList | None = None
    paragraphs_with_source_timestamps: list[Paragraph] | None = None
    enriched_article: Article | None = None


class PostProcessArticleNodes:
    def __init__(self, llm: InstructorLLM) -> None:
        self.llm = llm

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompt = read_text(f'{__package__}.resources.templates.post_process_article', f'{name}.txt')
        return prompt

    def add_additional_info_from_web(self, state: PostProcessArticleState) -> dict[str, WebSearchResultList]:
        prompt = self._load_prompt('add_additional_info_from_web').format(article=state.article.model_dump_json())
        queries = self.llm.get_completion(
            prompt,
            structured_output_type=WebSearchResultList,
            tools=['web_search'],
        )
        return {'web_search_data': queries}

    @staticmethod
    def get_referenced_timecodes(paragraph: Paragraph, transcript: Transcript) -> list[Timecodes | None]:
        paragraph_ref_ids = {seg.segment_id for seg in paragraph.segment_refs}
        return [segment.timecodes for segment in transcript.segments if segment.id in paragraph_ref_ids if
                segment.timecodes]

    def add_source_timestamps(self, state: PostProcessArticleState) -> dict[str, list[Paragraph]] | None:
        update: dict[str, list[Paragraph]] = {'paragraphs_with_source_timestamps': []}
        for paragraph in state.article.paragraphs:
            if not (referenced_timecodes := self.get_referenced_timecodes(paragraph, state.transcript)):
                update['paragraphs_with_source_timestamps'].append(paragraph)
                continue

            start_timecodes = [timecode.start_time for timecode in referenced_timecodes if timecode]
            end_timecodes = [timecode.end_time for timecode in referenced_timecodes if timecode]
            if not start_timecodes or not end_timecodes:
                update['paragraphs_with_source_timestamps'].append(paragraph)
                continue

            start_timecode = min(start_timecodes)
            end_timecode = max(end_timecodes)
            paragraph_with_timecodes = paragraph.model_copy(deep=True, update={
                'source_timecodes': Timecodes(start_time=start_timecode, end_time=end_timecode)})
            update['paragraphs_with_source_timestamps'].append(paragraph_with_timecodes)
        return update

    @staticmethod
    def enrich_article(state: PostProcessArticleState) -> dict[str, Article]:
        enriched_article = state.article.model_copy(deep=True)
        article_metadata = ArticleMetadata(
            web_searches=state.web_search_data,
        )
        enriched_article.metadata = article_metadata
        enriched_article.paragraphs = state.paragraphs_with_source_timestamps
        return {'enriched_article': enriched_article}

    def build_subgraph(self) -> CompiledStateGraph:
        builder = StateGraph(PostProcessArticleState)

        builder.add_node('add_additional_info_from_web', self.add_additional_info_from_web)
        builder.add_node('enrich_article', self.enrich_article)
        builder.add_node('add_source_timestamps', self.add_source_timestamps)

        builder.add_edge(START, 'add_additional_info_from_web')
        builder.add_edge(START, 'add_source_timestamps')
        builder.add_edge(['add_additional_info_from_web', 'add_source_timestamps'], 'enrich_article')
        builder.add_edge('enrich_article', END)

        return builder.compile()
