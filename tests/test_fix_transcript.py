from unittest.mock import Mock

import pytest

from jedireporter.fix_transcript import FixedState, FixTranscriptNodes
from jedireporter.transcript import (
    CorrectedText,
    Segment,
    SegmentBoundaryCorrection,
    SegmentSource,
    SegmentSplit,
    TextPart,
    Timecodes,
    Transcript
)


@pytest.fixture
def source_transcript() -> Transcript:
    return Transcript(
        id='src-1',
        language='en',
        segments=[SegmentSource(id=i, text=f'seg-{i}', speaker_id='spk1') for i in ['s1', 's2', 's3', 's4']],
        speakers=None,
    )


@pytest.mark.parametrize(
    ('segment_ids', 'expected_ids', 'raises'),
    [
        pytest.param({'s2'}, {'s2'}, None, id='returns_matching_segments'),
        pytest.param({'s1', 'missing'}, None, ValueError, id='raises_on_missing_segment'),
    ],
)
def test_extract_segments_by_ids(segment_ids: set[str], expected_ids: set[str] | None,
                                 raises: type[Exception] | None) -> None:
    segments = [Segment(id='s1', text='first', speaker_id='spk', segment_references={'s1'}),
                Segment(id='s2', text='second', speaker_id='spk', segment_references={'s2'})]
    if raises:
        with pytest.raises(raises):
            FixTranscriptNodes._extract_segments_by_ids(segments, segment_ids)
    else:
        extracted = FixTranscriptNodes._extract_segments_by_ids(segments, segment_ids)
        assert {segment.id for segment in extracted} == expected_ids


@pytest.mark.parametrize(
    ('segments', 'expected_ids', 'raises'),
    [
        pytest.param(
            [Segment(id='b', text='seg-b', speaker_id='spk', segment_references={'s4'}),
             Segment(id='c', text='seg-c', speaker_id='spk', segment_references={'s2', 's3'}),
             Segment(id='a', text='seg-a', speaker_id='spk', segment_references={'s1'})],
            ['a', 'c', 'b'],
            None,
            id='orders_by_source_reference',
        ),
        pytest.param(
            [Segment(id='bad', text='seg-bad', speaker_id='spk', segment_references={'sX'})],
            [None],
            ValueError,
            id='invalid_reference'
        )
    ]
)
def test_sort_segments(source_transcript: Transcript, segments: list[Segment], expected_ids: list[str],
                       raises: type[Exception] | None) -> None:
    state = FixedState(source=source_transcript)
    if raises:
        with pytest.raises(raises):
            FixTranscriptNodes._sort_segments(segments, state)
    else:
        sorted_segments = FixTranscriptNodes._sort_segments(segments, state)
        assert [s.id for s in sorted_segments] == expected_ids


@pytest.mark.parametrize(
    ('timecodes', 'text_parts', 'expected'),
    [
        pytest.param(
            None,
            [TextPart(text='hello', speaker_id='host'), TextPart(text='world', speaker_id='guest')],
            [None, None],
            id='no_timecodes',
        ),
        pytest.param(
            Timecodes(start_time=5, end_time=20),
            [
                TextPart(text='a', speaker_id='host'),
                TextPart(text='bb', speaker_id='host'),
                TextPart(text='ccc', speaker_id='guest'),
                TextPart(text='dddd', speaker_id='guest'),
                TextPart(text='eeeee', speaker_id='host'),
            ],
            [
                Timecodes(start_time=5, end_time=6),
                Timecodes(start_time=6, end_time=8),
                Timecodes(start_time=8, end_time=11),
                Timecodes(start_time=11, end_time=15),
                Timecodes(start_time=15, end_time=20),
            ],
            id='five_parts_proportional',
        ),
    ],
)
def test_get_timecodes_split(timecodes: Timecodes | None, text_parts: list[TextPart],
                             expected: list[Timecodes | None]) -> None:
    segment_split = SegmentSplit(id='seg', text_parts=text_parts)

    splits = FixTranscriptNodes._get_timecodes_split(timecodes=timecodes, segment_split=segment_split)
    assert splits == expected


def test_apply_segment_splits_splits_segment_and_propagates_timecodes() -> None:
    fixer = FixTranscriptNodes(llm=Mock())
    source_segments = [
        Segment(
            id='seg',
            text='abcde',
            speaker_id='host',
            timecodes=Timecodes(start_time=0, end_time=10),
            segment_references={'orig'},
        ),
        SegmentSource(id='keep', text='keep', speaker_id='host'),
    ]
    segment_splits = [
        SegmentSplit(
            id='seg',
            text_parts=[
                TextPart(text='abc', speaker_id='host'),
                TextPart(text='de', speaker_id='guest'),
            ],
        )
    ]
    split = fixer._apply_segment_splits(state=Mock(), segment_splits=segment_splits, segments=source_segments)

    assert [segment.id for segment in split] == ['seg_0', 'seg_1', 'keep']
    assert split[0].speaker_id == 'host'
    assert split[1].speaker_id == 'guest'
    assert split[0].segment_references == {'orig'}
    assert split[0].timecodes.start_time == pytest.approx(0)
    assert split[0].timecodes.end_time == pytest.approx(6)
    assert split[1].timecodes.start_time == pytest.approx(6)
    assert split[1].timecodes.end_time == pytest.approx(10)


def test_apply_grammar_fixes() -> None:
    segments = [
        Segment(id='s1', text='orig one', speaker_id='spk', segment_references={'orig'}),
        Segment(id='s2', text='orig two', speaker_id='spk', segment_references={'orig'}),
    ]
    corrected_texts = [CorrectedText(id='s1', corrected_text='fixed one')]

    fixed_segments = FixTranscriptNodes._apply_grammar_fixes(segments, corrected_texts)

    assert [segment.text for segment in fixed_segments] == ['fixed one', 'orig two']


@pytest.mark.parametrize(
    ('tc1', 'tc2', 'expected'),
    [
        pytest.param(None, None, None, id='both_none'),
        pytest.param(
            Timecodes(start_time=0, end_time=5),
            Timecodes(start_time=3, end_time=10),
            Timecodes(start_time=0, end_time=10),
            id='merges_overlap',
        ),
    ],
)
def test_merge_timecodes(tc1: Timecodes | None, tc2: Timecodes | None, expected: Timecodes | None) -> None:
    merged = FixTranscriptNodes._merge_timecodes(tc1, tc2)
    assert merged == expected


@pytest.mark.parametrize(
    ('text_a', 'text_b', 'expected'),
    [
        pytest.param('', 'second', 'second', id='missing_a'),
        pytest.param('first', '', 'first', id='missing_b'),
        pytest.param('first ', ' second', 'first second', id='joins_with_space'),
    ],
)
def test_merge_text(text_a: str, text_b: str, expected: str) -> None:
    assert FixTranscriptNodes._merge_text(text_a, text_b) == expected


def test_merge_same_speaker_neighbors() -> None:
    segments = [
        SegmentSource(id='s1', text='Hello', speaker_id='spk1', timecodes=Timecodes(start_time=0, end_time=1)),
        SegmentSource(id='s2', text='world', speaker_id='spk1', timecodes=Timecodes(start_time=1, end_time=2)),
        SegmentSource(id='s3', text='Next', speaker_id='spk2', timecodes=Timecodes(start_time=2, end_time=3)),
    ]
    mock_transcript = Mock(Transcript)
    mock_transcript.id = 'src'
    state = FixedState(source=mock_transcript, segments_with_fixed_boundaries_and_speakers=segments)
    fixer = FixTranscriptNodes(llm=Mock())

    update = fixer.merge_same_speaker_neighbors(state)

    merged = update['merged_segments']
    assert [segment.text for segment in merged] == ['Hello world', 'Next']
    assert merged[0].segment_references == {'s1', 's2'}
    assert merged[0].timecodes == Timecodes(start_time=0, end_time=2)
    assert merged[1].segment_references == {'s3'}
    assert merged[1].timecodes == Timecodes(start_time=2, end_time=3)


def test_set_timecodes_from_source() -> None:
    source = Transcript(
        id='src',
        language='en',
        segments=[
            SegmentSource(id='s1', text='first', speaker_id='spk', timecodes=Timecodes(start_time=1, end_time=2)),
            SegmentSource(id='s2', text='second', speaker_id='spk', timecodes=Timecodes(start_time=3, end_time=4)),
        ],
        speakers=None,
    )
    new_segments = [
        Segment(id='seg_0', text='merged', speaker_id='spk', timecodes=Timecodes(),
                segment_references={'s1', 's2'}),
    ]
    state = FixedState(source=source)

    updated = FixTranscriptNodes.set_timecodes_from_source(state, new_segments)

    assert updated[0].timecodes == Timecodes(start_time=1, end_time=4)


def test_apply_boundary_corrections() -> None:
    segments = [
        SegmentSource(id='s1', text='first', speaker_id='spk1'),
        SegmentSource(id='s2', text='second', speaker_id='spk2'),
    ]
    corrections = [
        SegmentBoundaryCorrection(id='s1', corrected_text='fixed', speaker_id=None),
        SegmentBoundaryCorrection(id='s2', corrected_text=None, speaker_id='spk3'),
    ]

    fixed = FixTranscriptNodes._apply_boundary_corrections(segments, corrections)

    assert [segment.text for segment in fixed] == ['fixed', 'second']
    assert [segment.speaker_id for segment in fixed] == ['spk1', 'spk3']


def test_apply_boundary_corrections_raises_on_unknown_id() -> None:
    segments = [SegmentSource(id='s1', text='first', speaker_id='spk1')]
    corrections = [SegmentBoundaryCorrection(id='missing', corrected_text='x', speaker_id=None)]

    with pytest.raises(ValueError):
        FixTranscriptNodes._apply_boundary_corrections(segments, corrections)
