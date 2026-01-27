from typing import Literal

from pydantic import AnyUrl, Field

from jedireporter.camelModel import CamelModel


class Timecodes(CamelModel):
    start_time: float | None = None
    end_time: float | None = None


class SegmentSource(CamelModel):
    id: str
    text: str
    speaker_id: str
    timecodes: Timecodes | None = None


class Segment(CamelModel):
    id: str = Field(
        description=(
            'Unique segment identifier. When merging segments during fixes, reuse the earliest contributing segment\'s '
            'ID.'
        )
    )
    text: str = Field(
        description=(
            'Spoken content for this segment (verbatim or lightly cleaned). Avoid fillers in fixed transcripts while '
            'preserving meaning.'
        )
    )
    speaker_id: str = Field(
        description=(
            'Identifier of the speaker who uttered this segment; must match an entry in `speakers`.'
        )
    )
    timecodes: Timecodes | None = Field(
        default=None,
        description=(
            'Optional timing for this segment. When merging, use earliest start and latest end of the merged range; '
            'never invent new times.'
        )
    )
    segment_references: set[str] = Field(
        description=(
            'If this segment was created from another (source) one, this field contains source segment IDs that this '
            'segment corresponds to (1:N mapping to original transcript).'
        )
    )

    @classmethod
    def from_source(cls, source_segment: SegmentSource, _id: str | None = None) -> 'Segment':
        return Segment(
            id=_id or source_segment.id,
            text=source_segment.text,
            speaker_id=source_segment.speaker_id,
            timecodes=source_segment.timecodes,
            segment_references={source_segment.id},
        )


class SegmentList(CamelModel):
    segments: list[Segment] = Field(description='List of segments.')


class TextPart(CamelModel):
    text: str = Field(
        description=(
            'Contiguous text fragment from the original segment after splitting around embedded '
            'conversational questions and/or mixed-speaker turns.'
        )
    )
    speaker_id: str = Field(
        description=(
            'Identifier of the speaker who most likely uttered this fragment; must match a '
            '`speaker_id` from the provided list of `Speaker`.'
        )
    )


class SegmentSplit(CamelModel):
    id: str = Field(
        description=(
            'Identifier of an original segment that is incorrectly merged (e.g., contains embedded '
            'conversational questions or mixed speaker fragments) and therefore requires splitting '
            'into multiple parts with different speakers.'
        )
    )
    text_parts: list[TextPart] = Field(
        description=(
            'Ordered list of text parts produced by splitting the segment around embedded '
            'conversational questions and/or mixed speaker fragments. Segments are included only '
            'when they contain at least two different speaker_ids, and adjacent TextPart entries '
            'must have different speaker_ids. Together, these parts must cover the entire original '
            'segment text.'
        ),
        min_length=2
    )


class SegmentSplitList(CamelModel):
    segment_splits: list[SegmentSplit] = Field(description='List of split segments.')


class RelevantSegmentIds(CamelModel):
    relevant_segments: set[str] = Field(description='Unique segment IDs, which are recognized as relevant for the given'
                                                    ' interview.')


class CorrectedText(CamelModel):
    id: str = Field(description='Unique segment identifier, which contains error (e.g., grammatical or verbal fillers) '
                                'and requires correction.')
    corrected_text: str = Field(description='Corrected text for this segment.')


class CorrectedTextList(CamelModel):
    corrected_texts: list[CorrectedText] = Field(description='List of corrected texts.')


class SegmentBoundaryCorrection(CamelModel):
    id: str = Field(description='Identifier of the segment that needs a boundary or speaker correction.')
    corrected_text: str | None = Field(
        default=None,
        description='Updated text for the segment after moving boundary words; None if text is unchanged.'
    )
    speaker_id: str | None = Field(
        default=None,
        description='Updated speaker_id when a segment was mislabeled; None if speaker is unchanged.'
    )


class SegmentBoundaryCorrectionList(CamelModel):
    corrections: list[SegmentBoundaryCorrection] = Field(description='List of boundary/speaker corrections.')


class Speaker(CamelModel):
    speaker_id: str = Field(
        description='Speaker identifier referenced by segments (e.g., "spk_0").'
    )
    role: Literal['host', 'guest', 'other'] = Field(
        description='Speaker role in the interview: "host" (interviewer), "guest" (interviewee) or '
                    '"other" (external, non-participant inserts).'
    )
    name: str | None = Field(
        default=None,
        description='Optional human-readable name of the speaker, if known from context.'
    )
    description: str | None = Field(
        default=None,
        description='Optional additional details about the speaker (e.g., title, affiliation).'
    )


class SpeakerList(CamelModel):
    speakers: list[Speaker] = Field(description='List of speakers.')


class Transcript(CamelModel):
    id: str = Field(
        description='Transcript identifier, stable across processing steps.'
    )
    language: str = Field(
        description='Language code of the transcript (e.g., "en").'
    )
    segments: list[Segment | SegmentSource] = Field(
        description='Ordered list of dialogue segments forming the interview.'
    )
    speakers: list[Speaker] | None = Field(
        default=None,
        description=(
            'Optional list of declared speakers used in segments. Ensure each `speaker_id` in segments is declared '
            'here.'
        )
    )
    url: AnyUrl | None = Field(
        default=None,
        description='Optional URL to the source audio/video from which the transcript was produced.'
    )
