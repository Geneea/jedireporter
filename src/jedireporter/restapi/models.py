"""
REST API models for the JEDI Reporter service.

This module defines API-specific models that are decoupled from internal processing models.
These models represent the public API contract and can evolve independently from internal implementation.
"""
from typing import Literal

from pydantic import AnyUrl, Field

from jedireporter.article import (
    Article,
    ArticleMetadata,
    Paragraph,
    SegmentRef,
    TextOccurrence,
    WebSearchResult,
    WebSource,
)
from jedireporter.camelModel import CamelModel
from jedireporter.transcript import Segment, SegmentSource, Speaker, Timecodes, Transcript


class APITimecodes(CamelModel):
    """
    Timing information for a transcript segment.

    Specifies the start and end time of a segment in seconds from the beginning
    of the audio/video source.
    """
    start_time: float = Field(
        description="Start time of the segment in seconds from the beginning of the recording"
    )
    end_time: float = Field(
        description="End time of the segment in seconds from the beginning of the recording"
    )

    def to_internal(self) -> Timecodes:
        """Convert API model to internal Timecodes model."""
        return Timecodes(
            start_time=self.start_time,
            end_time=self.end_time
        )

    @classmethod
    def from_internal(cls, timecodes: Timecodes) -> 'APITimecodes':
        """Convert internal Timecodes model to API model."""
        return cls(
            start_time=timecodes.start_time,
            end_time=timecodes.end_time
        )


class APISpeaker(CamelModel):
    """
    A participant in an interview.

    Contains speaker identification, role classification (host, guest, or other),
    and optional metadata such as name and description.
    """
    speaker_id: str = Field(
        description='Unique speaker identifier referenced in segments (e.g., "spk_0", "host", "guest")'
    )
    role: Literal['host', 'guest', 'other'] = Field(
        description='Speaker role: "host" (interviewer), "guest" (interviewee), or "other" (narrator, external voice)'
    )
    name: str | None = Field(
        default=None,
        description='Optional human-readable name of the speaker (e.g., "Jane Doe")'
    )
    description: str | None = Field(
        default=None,
        description='Optional additional details about the speaker (e.g., title, affiliation, credentials)'
    )

    def to_internal(self) -> Speaker:
        """Convert API model to internal Speaker model."""
        return Speaker(
            speaker_id=self.speaker_id,
            role=self.role,
            name=self.name,
            description=self.description
        )

    @classmethod
    def from_internal(cls, speaker: Speaker) -> 'APISpeaker':
        """Convert internal Speaker model to API model."""
        return cls(
            speaker_id=speaker.speaker_id,
            role=speaker.role,
            name=speaker.name,
            description=speaker.description
        )


class APISegment(CamelModel):
    """
    A single dialogue segment in the transcript.

    Represents one continuous utterance by a speaker, optionally with timing information.
    Segments are ordered sequentially to form the complete interview transcript.
    """
    id: str = Field(
        description="Unique identifier for this segment within the transcript"
    )
    text: str = Field(
        description="The spoken content of this segment (verbatim or lightly edited for readability)"
    )
    speaker_id: str = Field(
        description="ID of the speaker who uttered this segment (must match a speaker_id in the speakers list)"
    )
    timecodes: APITimecodes | None = Field(
        default=None,
        description="Optional timing information indicating when this segment was spoken"
    )

    def to_internal(self) -> SegmentSource:
        """Convert API model to internal Segment model."""
        return SegmentSource(
            id=self.id,
            text=self.text,
            speaker_id=self.speaker_id,
            timecodes=self.timecodes.to_internal() if self.timecodes else None,
            # API input never includes internal segment references
        )

    @classmethod
    def from_internal(cls, segment: Segment) -> 'APISegment':
        """Convert internal Segment model to API model."""
        return cls(
            id=segment.id,
            text=segment.text,
            speaker_id=segment.speaker_id,
            timecodes=APITimecodes.from_internal(segment.timecodes) if segment.timecodes else None
            # Note: segment_references is internal-only and not exposed in API
        )


class APITranscript(CamelModel):
    """
    A transcribed interview.

    Contains dialogue segments with speaker attribution, optional speaker metadata,
    and optional timing information from the source audio/video recording.
    """
    id: str = Field(
        description="Unique identifier for this transcript (used for tracking and result correlation)"
    )
    language: str = Field(
        description=(
            'ISO 639-1 language code of the transcript content '
            '(e.g., "en" for English, "cs" for Czech, "de" for German)'
        )
    )
    segments: list[APISegment] = Field(
        description="Ordered list of dialogue segments that form the complete interview transcript"
    )
    speakers: list[APISpeaker] | None = Field(
        default=None,
        description=(
            "Optional list of speakers participating in the interview. "
            "If provided, each speaker_id used in segments must be declared here"
        )
    )
    url: AnyUrl | None = Field(
        default=None,
        description="Optional URL pointing to the source audio or video recording"
    )

    def to_internal(self) -> Transcript:
        """Convert API model to internal Transcript model."""
        return Transcript(
            id=self.id,
            language=self.language,
            segments=[seg.to_internal() for seg in self.segments],
            speakers=[spk.to_internal() for spk in self.speakers] if self.speakers else None,
            url=self.url
        )

    @classmethod
    def from_internal(cls, transcript: Transcript) -> 'APITranscript':
        return cls(
            id=transcript.id,
            language=transcript.language,
            segments=[APISegment.from_internal(seg) for seg in transcript.segments],
            speakers=[APISpeaker.from_internal(spk) for spk in transcript.speakers] if transcript.speakers else None,
            url=transcript.url
        )


class APISegmentRef(CamelModel):
    """
    A reference to a transcript segment or a text span within it.

    Points to a whole segment by ID, or to a specific character range within
    a segment using optional offset and length fields.
    """
    segment_id: str = Field(
        description="ID of the transcript segment being referenced"
    )
    offset: int | None = Field(
        default=None,
        description="Optional character offset (0-based) indicating the start position within the segment text"
    )
    length: int | None = Field(
        default=None,
        description="Optional length in characters of the referenced text span (used with offset)"
    )

    @classmethod
    def from_internal(cls, ref: SegmentRef) -> 'APISegmentRef':
        """Convert internal SegmentRef model to API model."""
        return cls(
            segment_id=ref.segment_id,
            offset=ref.offset,
            length=ref.length
        )


class APIWebSource(CamelModel):
    """
    Source citation for web-extracted context.

    Captures the source URL and snippet metadata for traceability.
    """
    url: str = Field(description='Source URL supporting the editorial note.')
    title: str = Field(description='Title of the source page.')
    snippet: str = Field(description='Short excerpt from the source that supports the note.')

    @classmethod
    def from_internal(cls, source: WebSource) -> 'APIWebSource':
        return cls(
            url=source.url,
            title=source.title,
            snippet=source.snippet,
        )


class APITextOccurrence(CamelModel):
    """
    Location of a subject mention in the article text.

    Links a surface form to the paragraph where it appears.
    """
    paragraph_id: str = Field(
        description='Identifier of the paragraph containing this mention.'
    )
    text_string: str = Field(
        description='Exact text as it appears in the paragraph.'
    )

    @classmethod
    def from_internal(cls, occurrence: TextOccurrence) -> 'APITextOccurrence':
        return cls(
            paragraph_id=occurrence.paragraph_id,
            text_string=occurrence.text_string,
        )


class APIWebSearchResult(CamelModel):
    """
    Web-derived context for a single subject in the article.

    Includes the search query, summary, sources, and in-article mentions.
    """
    id: str = Field(
        description='Stable identifier for this subject (slug-like, lowercase, hyphen-separated).'
    )
    text_occurrences: list[APITextOccurrence] = Field(
        description='Mentions in the article that refer to this subject.'
    )
    query: str = Field(
        description='Web search query used to retrieve context for this subject.'
    )
    summary: str = Field(
        description='Short factual summary explaining who/what the subject is and why it is relevant.'
    )
    sources: list[APIWebSource] = Field(description='Sources used to craft the summary.')

    @classmethod
    def from_internal(cls, data: WebSearchResult) -> 'APIWebSearchResult':
        return cls(
            id=data.id,
            text_occurrences=[APITextOccurrence.from_internal(item) for item in data.text_occurrences],
            query=data.query,
            summary=data.summary,
            sources=[APIWebSource.from_internal(source) for source in data.sources]
        )


class APIArticleMetadata(CamelModel):
    """
    Metadata attached to an article.

    Holds optional enrichment details such as web-derived context.
    """
    web_searches: list[APIWebSearchResult] | None = Field(
        default=None,
        description='Optional web-derived context for entities mentioned in the article.'
    )

    @classmethod
    def from_internal(cls, metadata: ArticleMetadata) -> 'APIArticleMetadata':
        return cls(
            web_searches=[APIWebSearchResult.from_internal(web_search) for web_search in metadata.web_searches.data]
            if metadata.web_searches else None
        )


class APIParagraph(CamelModel):
    """
    A paragraph of an article.

    Contains the paragraph type (title, lead, text, question, answer, etc.),
    the text content, optional speaker attribution, and references to source
    transcript segments.
    """
    id: str = Field(
        description='Unique identifier for this paragraph within the article'
    )
    type: Literal['title', 'lead', 'summary', 'text', 'question', 'answer', 'conclusion'] = Field(
        description=(
            'Paragraph type: '
            '"title" (article headline or section heading), '
            '"lead" (article lead or abstract), '
            '"summary" (article text summary), '
            '"text" (narrative, introduction, or context), '
            '"question" (interviewer question), '
            '"answer" (interviewee response), '
            '"conclusion" (closing remarks)'
        )
    )
    text: str = Field(
        description="Content of the paragraph"
    )
    speaker: str | None = Field(
        default=None,
        description=(
            'Speaker ID for "question" and "answer" paragraphs (must match a speaker_id from the transcript). '
            'Should be null for "title" and "text" paragraphs'
        )
    )
    segment_refs: list[APISegmentRef] = Field(
        description="References to the source transcript segments that support or inspired this paragraph"
    )
    source_timecodes: APITimecodes | None = Field(
        default=None,
        description='Optional timing information indicating when this paragraph was spoken in the source audio.'
    )

    @classmethod
    def from_internal(cls, paragraph: Paragraph) -> 'APIParagraph':
        """Convert internal Paragraph model to API model."""
        return cls(
            id=paragraph.id,
            type=paragraph.type,
            text=paragraph.text,
            speaker=paragraph.speaker,
            segment_refs=[APISegmentRef.from_internal(ref) for ref in paragraph.segment_refs],
            source_timecodes=APITimecodes.from_internal(
                paragraph.source_timecodes) if paragraph.source_timecodes else None
        )


class APIArticle(CamelModel):
    """
    A structured article derived from an interview transcript.

    Contains an ordered sequence of paragraphs (titles, questions, answers, narrative text)
    along with the source transcript for traceability.
    """
    id: str = Field(
        description=(
            'Article identifier (typically "article-" prefix '
            'followed by the transcript ID)'
        )
    )
    language: str = Field(
        description="ISO 639-1 language code matching the source transcript"
    )
    paragraphs: list[APIParagraph] = Field(
        description=(
            "Ordered list of paragraphs forming the complete article "
            "(titles, questions, answers, narrative text)"
        )
    )
    transcript: APITranscript = Field(
        description="The source transcript that was used to generate this article (for full traceability)"
    )
    metadata: APIArticleMetadata | None = Field(
        default=None,
        description='Optional article metadata including web-derived context.'
    )

    @classmethod
    def from_internal(cls, article: Article) -> 'APIArticle':
        """
        Convert internal Article model to API model.

        Note: The article.transcript field must be set before calling this method,
        as it's required in the API response for traceability.
        """
        if article.transcript is None:
            raise ValueError("Article must have transcript set for API conversion")

        # Convert transcript to API format
        api_transcript = APITranscript(
            id=article.transcript.id,
            language=article.transcript.language,
            segments=[APISegment.from_internal(seg) for seg in article.transcript.segments],
            speakers=[APISpeaker.from_internal(spk) for spk in article.transcript.speakers]
            if article.transcript.speakers else None,
            url=article.transcript.url
        )

        return cls(
            id=article.id,
            language=article.language,
            paragraphs=[APIParagraph.from_internal(p) for p in article.paragraphs],
            transcript=api_transcript,
            metadata=APIArticleMetadata.from_internal(article.metadata) if article.metadata else None,
            # Note: debug_info is internal-only and not exposed in API
        )
