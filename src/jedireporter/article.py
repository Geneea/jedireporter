from typing import Literal

from pydantic import Field, model_validator

from jedireporter.camelModel import CamelModel
from jedireporter.transcript import Timecodes, Transcript


class Topic(CamelModel):
    """
    A discussed topic detected in the interview.

    The `id` should be a stable, slug-like identifier derived from the title (lowercase, hyphens, no spaces)
    and is used by paragraphs to reference the topic.
    """
    id: str = Field(description='Stable identifier for this topic (e.g., "project-launch", "ai-ethics").')
    title: str = Field(description='Human-readable title of the topic to display in exports.')
    description: str | None = Field(default=None, description='Optional short description of the topic.')


class TopicList(CamelModel):
    topics: list[Topic] = Field(description='List of topics detected in the interview.')


class SegmentRef(CamelModel):
    """
    Reference to a character span within a transcript segment.

    ``segment_id`` points to a segment in the fixed transcript; offset and length (0-based) select a
    substring of that segment's text. To cite a whole segment, use offset=0 and length=len(text).
    """
    segment_id: str = Field(
        description=(
            'ID of the transcript segment this reference points to.'
        )
    )
    offset: int | None = None
    length: int | None = None


class Paragraph(CamelModel):
    id: str | None = Field(
        description=(
            'Identifier for this paragraph. Used by downstream annotations such as '
            'web search queries. LLM leaves this field `None` and the ID is generated afterwards automatically'
            ' from the paragraph\'s position within the article.'
        )
    )
    type: Literal['title', 'lead', 'summary', 'text', 'question', 'answer', 'conclusion'] = Field(
        description=(
            'Paragraph type: "title" (headline), "summary" (3–5 key points summary following the title), '
            '"text" (narration/intro/context), "question" (interviewer prompt), or "answer" (interviewee response).'
        )
    )
    text: str = Field(
        description=(
            'Paragraph content. For "answer", keep the interviewee\'s words faithful to the transcript; '
            'only minimal grammar fixes are acceptable. For "question", wording may be changed for flow.'
        )
    )
    speaker: str | None = Field(
        default=None,
        description=(
            'Speaker assignment for this paragraph. For "question" and "answer", set to a valid '
            'speaker_id from Transcript.speakers; derive primarily from the speakers of the referenced segments, '
            'resolving inconsistencies using full interview context if needed. For "title" and "text", leave null.'
        )
    )
    segment_refs: list[SegmentRef] = Field(
        description=(
            'List of spans in the transcript (input to this step) that support this paragraph. Each reference points to'
            ' a segment ID. If the segments themselves carry references to the original "source" segments in the field'
            ' `segment_references`, **do not reference those here**. For editorial questions, reference nearby answer '
            'segments to anchor placement.'
        )
    )
    topic_id: str | None = Field(
        default=None,
        description=(
            'Optional reference to a Topic.id. Only allowed for Q&A paragraphs (type=="question", "answer" or "text").'
        )
    )
    source_timecodes: Timecodes | None = None

    @model_validator(mode='after')
    def check_speaker(self) -> 'Paragraph':
        if self.type in ('answer', 'question') and self.speaker is None:
            raise ValueError(f'Speaker ID {self.speaker} is required when the type is {self.type}')
        return self

    @model_validator(mode='after')
    def check_topic_id(self) -> 'Paragraph':
        if self.type not in ('answer', 'question', 'text') and self.topic_id is not None:
            raise ValueError('topic_id is only allowed for question/answer/text paragraphs')
        elif self.type in ('answer', 'question', 'text') and self.topic_id is None:
            raise ValueError('topic_id is required for question/answer/text paragraphs')
        return self


class WebSource(CamelModel):
    url: str = Field(description='Source URL.')
    start_index: int | None = Field(
        default=None,
        description='Start index of the cited span within the source, if available.'
    )
    end_index: int | None = Field(
        default=None,
        description='End index of the cited span within the source, if available.'
    )
    title: str = Field(description='Webpage title of the source.')
    snippet: str = Field(
        description='Short excerpt or snippet from the source that supports the note.'
    )


class TextOccurrence(CamelModel):
    paragraph_id: str = Field(
        description=(
            'Identifier of the paragraph containing this occurrence. Must match the paragraph\'s `id`.'
        )
    )
    text_string: str = Field(
        description=(
            'Exact surface form from the paragraph (keep each distinct spelling/variant as a separate occurrence).'
        )
    )


class WebSearchResult(CamelModel):
    id: str = Field(
        description=(
            'Stable identifier for this query (slug-like, lowercase, hyphen-separated, derived from the subject).'
        )
    )
    text_occurrences: list[TextOccurrence] = Field(
        description=(
            'All mentions in the article that refer to the same subject needing extra context; include every variant '
            'string.'
        )
    )
    query: str = Field(
        description=(
            'Exact web search query to retrieve a short, factual explainer about the subject; include disambiguating '
            'details.'
        )
    )
    summary: str = Field(
        description=(
            '2-3 sentence factual note derived from web search results explaining who/what the subject is and why it '
            'matters.'
        )
    )
    sources: list[WebSource] = Field(
        description=(
            'List of sources used to craft the editorial note.'
        )
    )

    @model_validator(mode='after')
    def check_sources(self) -> 'WebSearchResult':
        if not self.sources:
            raise ValueError('At least one source must be provided.')
        return self

    @model_validator(mode='after')
    def check_summary(self) -> 'WebSearchResult':
        if not self.summary:
            raise ValueError('A summary must be provided.')
        return self


class WebSearchResultList(CamelModel):
    data: list[WebSearchResult] = Field(
        description='Collection of web search queries capturing subjects in the article that need additional '
                    'background.'
    )


class ArticleMetadata(CamelModel):
    """Stores enrichment data for the article, such as extracted info from web, etc."""
    web_searches: WebSearchResultList | None = Field(
        default=None,
        description='Collection of extracted data for proper nouns, which require additional context, from web.')


class Article(CamelModel):
    id: str = Field(
        description=(
            'Article identifier. Reuse the source transcript ID prefixed with "article-" when generating.'
        )
    )
    language: str = Field(
        description=(
            'Language code of the article (copy from the transcript), e.g., "en".'
        )
    )
    paragraphs: list[Paragraph] = Field(
        description=(
            'Ordered list of paragraphs forming the final article: headline(s), intro, Q&A, and conclusion.'
        )
    )
    topics: TopicList = Field(description='List of detected topics for this article/interview.')
    transcript: Transcript | None = None
    metadata: ArticleMetadata | None = None
    debug_info: dict[str, str] | None = None
