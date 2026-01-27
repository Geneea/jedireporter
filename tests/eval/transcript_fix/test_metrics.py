from importlib.resources import open_text
from typing import Any, Literal

import numpy as np
import pytest

from jedireporter.eval.transcript_fix.metrics import TranscriptMetrics
from jedireporter.eval.transcript_fix.models import (
    SegmentationMetrics,
    SpeakerMetrics,
    TextMetricsSegment,
    TextMetricsTranscript,
    TimingMetrics,
)
from jedireporter.eval.transcript_fix.utils import TranscriptPair
from jedireporter.transcript import (
    Segment,
    Speaker,
    Timecodes,
    Transcript,
)


class TestTranscriptMetrics:
    @pytest.fixture(scope='class')
    def source_transcript(self) -> Transcript:
        with open_text('tests.resources', 'test_source_transcript.jsonl') as f:
            return Transcript.model_validate_json(f.readline())

    @pytest.fixture(scope='class')
    def gold_transcript(self) -> Transcript:
        with open_text('tests.resources', 'test_gold_transcript.jsonl') as f:
            return Transcript.model_validate_json(f.readline())

    @pytest.fixture
    def transcript_pair(self, source_transcript: Transcript,
                        gold_transcript: Transcript) -> TranscriptPair:
        return TranscriptPair(source=source_transcript, gold=gold_transcript, candidate=gold_transcript)

    @pytest.fixture
    def transcript_metrics(self, transcript_pair: TranscriptPair) -> TranscriptMetrics:
        return TranscriptMetrics(transcript_pair)

    def test_join_transcript(self, transcript_metrics: TranscriptMetrics, source_transcript: Transcript):
        expected_text = ('It\'s Sunday morning on CBS, and here again is Jane Pauley. For the past 6 decades, whether '
                         'he\'s directing, writing, or acting, the award winning filmmaker Werner Herzog has been a '
                         'most singular voice in the movies. This morning he\'s using that voice in conversation with '
                         'our Ben Mankiewicz. Uh, Werner, did you, uh, turn your cell phone off?')
        result = transcript_metrics._join_transcript(source_transcript)
        assert result == expected_text

    def test_text_metrics_transcript_level(self, transcript_metrics: TranscriptMetrics) -> None:
        result = transcript_metrics._text_metrics_transcript_level()
        expected_outcome = TextMetricsTranscript(
            candidate2source_wer=0.11864406779661017,
            candidate_word_count=56,
            source_word_count=59,
            candidate2gold_wer=0.0,
            source2gold_wer=0.125,
            gold_word_count=56,
        )
        assert result == expected_outcome

    def test_concatenate_source_segments(self,
                                         transcript_metrics: TranscriptMetrics,
                                         source_transcript: Transcript,
                                         gold_transcript: Transcript) -> None:
        source_by_id: dict[str, Segment] = {segment.id: segment for segment in source_transcript.segments}
        source_order: dict[str, int] = {segment.id: idx for idx, segment in enumerate(source_transcript.segments)}

        gold_segment = gold_transcript.segments[0]
        source_text, used_source_ids = transcript_metrics._concatenate_source_segments(gold_segment, source_order,
                                                                                       source_by_id, gold_transcript)
        assert source_text == (' It\'s Sunday morning on CBS, and here again is Jane Pauley. For the past 6 decades, '
                               'whether he\'s directing, writing, or acting, the award winning filmmaker Werner Herzog '
                               'has been a most singular voice in the movies. This morning he\'s using that voice in '
                               'conversation with our Ben Mankiewicz.')
        assert used_source_ids == {'2', '3'}

    def test_text_metrics_segment_level(self, transcript_metrics: TranscriptMetrics) -> None:
        result = transcript_metrics._text_metrics_segment_level()
        expected_outcome = TextMetricsSegment(
            candidate_segment_count=2,
            source_segment_count=4,
            omitted_source_segment_count=0,
            average_wer=0.19081632653061223,
            max_wer=0.3,
            max_wer_segment_id='Test_transcript.mp3: 4',
        )
        assert result == expected_outcome

    def test_segmentation_metrics(self, transcript_metrics: TranscriptMetrics) -> None:
        result = transcript_metrics._segmentation_metrics()

        expected_outcome = SegmentationMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            candidate_segment_count=2,
            gold_segment_count=2,
        )

        assert result == expected_outcome

    @pytest.fixture
    def speaker_assignment_pair(self) -> TranscriptPair:
        # gold speakers spk_gold_1 (0-5), spk_gold_2 (5-10)
        gold = Transcript(
            id='gold',
            language='en',
            segments=[Segment(id='seg_1', text='SEG_1', speaker_id='spk_gold_1',
                              timecodes=Timecodes(start_time=0.0, end_time=5.0),
                              segment_references=set()),
                      Segment(id='seg_2', text='SEG_2', speaker_id='spk_gold_2',
                              timecodes=Timecodes(start_time=5.0, end_time=10.0),
                              segment_references=set())])
        # candidate spk_can_1 (0-6), spk_can_2 (6-10)
        candidate = Transcript(
            id='candidate',
            language='en',
            segments=[Segment(id='seg_1', text='SEG_1', speaker_id='spk_can_1',
                              timecodes=Timecodes(start_time=0.0, end_time=6.0),
                              segment_references=set()),
                      Segment(id='seg_2', text='SEG_2', speaker_id='spk_can_2',
                              timecodes=Timecodes(start_time=6.0, end_time=10.0),
                              segment_references=set())])
        return TranscriptPair(gold=gold, candidate=candidate, source=candidate)

    @staticmethod
    def _transcript_from_specs(transcript_id: str, specs: list[dict[str, str | float | None]]) -> Transcript:
        segments: list[Segment] = []
        for spec in specs:
            start = spec.get('start')
            end = spec.get('end')
            timecodes = None if start is None or end is None else Timecodes(start_time=start, end_time=end)
            segments.append(
                Segment(
                    id=spec['id'],
                    text='',
                    speaker_id=spec['speaker'],
                    timecodes=timecodes,
                    segment_references=spec.get('segment_references') or set(),
                )
            )
        return Transcript(id=transcript_id, language='en', segments=segments)

    @pytest.mark.parametrize(
        ('gold_specs', 'candidate_specs', 'expected_gold_ids', 'expected_candidate_ids', 'expected_matrix'),
        [
            pytest.param(
                [
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': 0.0, 'end': 5.0},
                    {'id': 'seg_2', 'speaker': 'spk_gold_2', 'start': 5.0, 'end': 10.0},
                ],
                [
                    {'id': 'seg_1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 6.0},
                    {'id': 'seg_2', 'speaker': 'spk_can_2', 'start': 6.0, 'end': 10.0},
                ],
                ['spk_gold_1', 'spk_gold_2'],
                ['spk_can_1', 'spk_can_2'],
                np.array([[5.0, 0.0], [1.0, 4.0]], dtype=float),
                id='baseline-two-speakers',
            ),
            pytest.param(
                [
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': 0.0, 'end': 4.0},
                    {'id': 'seg_2', 'speaker': 'spk_gold_2', 'start': 4.0, 'end': 9.0},
                    {'id': 'seg_3', 'speaker': 'spk_gold_3', 'start': 9.0, 'end': 14.0},
                    {'id': 'seg_4', 'speaker': 'spk_gold_4', 'start': 14.0, 'end': 20.0},
                ],
                [
                    {'id': 'seg_c1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 5.0},
                    {'id': 'seg_c2', 'speaker': 'spk_can_2', 'start': 5.0, 'end': 10.0},
                    {'id': 'seg_c3', 'speaker': 'spk_can_3', 'start': 9.0, 'end': 16.0},
                    {'id': 'seg_c4', 'speaker': 'spk_can_4', 'start': 16.0, 'end': 20.0},
                ],
                ['spk_gold_1', 'spk_gold_2', 'spk_gold_3', 'spk_gold_4'],
                ['spk_can_1', 'spk_can_2', 'spk_can_3', 'spk_can_4'],
                np.array(
                    [
                        [4.0, 0.0, 0.0, 0.0],
                        [1.0, 4.0, 0.0, 0.0],
                        [0.0, 1.0, 5.0, 0.0],
                        [0.0, 0.0, 2.0, 4.0],
                    ],
                    dtype=float,
                ),
                id='larger-matrix-with-multi-overlap',
            ),
            pytest.param(
                [
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': 0.0, 'end': 2.0},
                    {'id': 'seg_2', 'speaker': 'spk_gold_2', 'start': 2.0, 'end': 4.0},
                    {'id': 'seg_3', 'speaker': 'spk_gold_3', 'start': 4.0, 'end': 8.0},
                ],
                [
                    {'id': 'seg_c1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 5.0},
                    {'id': 'seg_c2', 'speaker': 'spk_can_2', 'start': 5.0, 'end': 8.0},
                ],
                ['spk_gold_1', 'spk_gold_2', 'spk_gold_3'],
                ['spk_can_1', 'spk_can_2'],
                np.array(
                    [
                        [2.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [1.0, 3.0, 0.0],
                    ],
                    dtype=float,
                ),
                id='gold-extra-speaker-padding',
            ),
            pytest.param(
                [
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': 0.0, 'end': 4.0},
                    {'id': 'seg_2', 'speaker': 'spk_gold_2', 'start': 4.0, 'end': 8.0},
                ],
                [
                    {'id': 'seg_c1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 4.0},
                    {'id': 'seg_c2', 'speaker': 'spk_can_2', 'start': 4.0, 'end': 6.0},
                    {'id': 'seg_c3', 'speaker': 'spk_can_3', 'start': 6.0, 'end': 10.0},
                ],
                ['spk_gold_1', 'spk_gold_2'],
                ['spk_can_1', 'spk_can_2', 'spk_can_3'],
                np.array(
                    [
                        [4.0, 0.0, 0.0],
                        [0.0, 2.0, 2.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                id='candidate-extra-speaker-padding',
            ),
            pytest.param(
                [
                    {'id': 'seg_0', 'speaker': 'spk_gold_0', 'start': None, 'end': None},
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': 0.0, 'end': 3.0},
                    {'id': 'seg_2', 'speaker': 'spk_gold_2', 'start': 3.0, 'end': 6.0},
                ],
                [
                    {'id': 'seg_c0', 'speaker': 'spk_can_0', 'start': None, 'end': None},
                    {'id': 'seg_c1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 2.0},
                    {'id': 'seg_c2', 'speaker': 'spk_can_2', 'start': 2.0, 'end': 6.0},
                ],
                ['spk_gold_1', 'spk_gold_2'],
                ['spk_can_1', 'spk_can_2'],
                np.array(
                    [
                        [2.0, 1.0],
                        [0.0, 3.0],
                    ],
                    dtype=float,
                ),
                id='partial-missing-timecodes',
            ),
            pytest.param(
                [
                    {'id': 'seg_1', 'speaker': 'spk_gold_1', 'start': None, 'end': None},
                ],
                [
                    {'id': 'seg_1', 'speaker': 'spk_can_1', 'start': 0.0, 'end': 1.0},
                ],
                [],
                ['spk_can_1'],
                np.zeros((0, 0), dtype=float),
                id='no-gold-timecodes',
            ),
        ],
    )
    def test_build_speaker_overlap_matrix(
        self,
        gold_specs: list[dict[str, str | float | None]],
        candidate_specs: list[dict[str, str | float | None]],
        expected_gold_ids: list[str],
        expected_candidate_ids: list[str],
        expected_matrix: np.ndarray,
    ) -> None:
        gold = self._transcript_from_specs('gold', gold_specs)
        candidate = self._transcript_from_specs('candidate', candidate_specs)
        pair = TranscriptPair(gold=gold, candidate=candidate, source=candidate)
        transcript_metrics = TranscriptMetrics(pair=pair)

        matrix, gold_ids, candidate_ids = transcript_metrics._build_speaker_overlap_matrix(gold, candidate)

        assert gold_ids == expected_gold_ids
        assert candidate_ids == expected_candidate_ids
        assert matrix.shape == expected_matrix.shape
        np.testing.assert_allclose(matrix, expected_matrix)

    @staticmethod
    def _transcript_from_reference_specs(
        transcript_id: str,
        speaker_specs: list[dict[str, Literal['host', 'guest', 'other']]],
        segment_specs: list[dict[str, str | set[str]]],
    ) -> Transcript:
        speakers = [
            Speaker(speaker_id=spec['speaker_id'], role=spec.get('role', 'other')) for spec in speaker_specs
        ]
        segments: list[Segment] = []
        for spec in segment_specs:
            segments.append(
                Segment(
                    id=spec['id'],
                    text='',
                    speaker_id=spec['speaker'],
                    segment_references=set(spec['segment_references']),
                )
            )
        return Transcript(id=transcript_id, language='en', segments=segments, speakers=speakers)

    @pytest.mark.parametrize(
        ('gold_spec', 'candidate_spec', 'expected_matrix'),
        [
            pytest.param(
                {
                    'speakers': [
                        {'speaker_id': 'spk_gold_1', 'role': 'host'},
                        {'speaker_id': 'spk_gold_2', 'role': 'guest'},
                    ],
                    'segments': [
                        {'id': 'g1', 'speaker': 'spk_gold_1', 'segment_references': {'src_1'}},
                        {'id': 'g2', 'speaker': 'spk_gold_1', 'segment_references': {'src_2'}},
                        {'id': 'g3', 'speaker': 'spk_gold_2', 'segment_references': {'src_3'}},
                    ],
                },
                {
                    'speakers': [
                        {'speaker_id': 'spk_can_1', 'role': 'host'},
                        {'speaker_id': 'spk_can_2', 'role': 'guest'},
                    ],
                    'segments': [
                        {'id': 'c1', 'speaker': 'spk_can_1', 'segment_references': {'src_1'}},
                        {'id': 'c2', 'speaker': 'spk_can_1', 'segment_references': {'src_3'}},
                        {'id': 'c3', 'speaker': 'spk_can_2', 'segment_references': {'src_2'}},
                    ],
                },
                np.array([[1.0, 1.0], [1.0, 0.0]], dtype=float),
                id='baseline-overlap-count',
            ),
            pytest.param(
                {
                    'speakers': [
                        {'speaker_id': 'spk_gold_1', 'role': 'host'},
                        {'speaker_id': 'spk_gold_2', 'role': 'guest'},
                    ],
                    'segments': [
                        {'id': 'g1', 'speaker': 'spk_gold_1', 'segment_references': {'src_1'}},
                        {'id': 'g2', 'speaker': 'spk_gold_1', 'segment_references': {'src_2'}},
                        {'id': 'g3', 'speaker': 'spk_gold_2', 'segment_references': {'src_3'}},
                    ],
                },
                {
                    'speakers': [
                        {'speaker_id': 'spk_can_1', 'role': 'host'},
                        {'speaker_id': 'spk_can_2', 'role': 'guest'},
                        {'speaker_id': 'spk_can_3', 'role': 'guest'},
                    ],
                    'segments': [
                        {'id': 'c1', 'speaker': 'spk_can_1', 'segment_references': {'src_1'}},
                        {'id': 'c2', 'speaker': 'spk_can_2', 'segment_references': {'src_2'}},
                        {'id': 'c3', 'speaker': 'spk_can_2', 'segment_references': {'src_3'}},
                        {'id': 'c4', 'speaker': 'spk_can_3', 'segment_references': {'src_5'}},
                    ],
                },
                np.array(
                    [
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                id='candidate-extra-speaker-padding',
            ),
        ],
    )
    def test_build_speaker_overlap_matrix_no_timecodes(
        self,
        gold_spec: dict[str, list[dict[str, Any]]],
        candidate_spec: dict[str, list[dict[str, Any]]],
        expected_matrix: np.ndarray,
    ) -> None:
        gold = self._transcript_from_reference_specs('gold', gold_spec['speakers'], gold_spec['segments'])
        candidate = self._transcript_from_reference_specs(
            'candidate', candidate_spec['speakers'], candidate_spec['segments']
        )
        pair = TranscriptPair(gold=gold, candidate=candidate, source=candidate)
        transcript_metrics = TranscriptMetrics(pair=pair)

        matrix, gold_ids, candidate_ids = transcript_metrics._build_speaker_overlap_matrix_no_timecodes(gold, candidate)

        expected_gold_ids = [spec['speaker_id'] for spec in gold_spec['speakers']]
        expected_candidate_ids = [spec['speaker_id'] for spec in candidate_spec['speakers']]

        assert gold_ids == expected_gold_ids
        assert candidate_ids == expected_candidate_ids
        assert matrix.shape == expected_matrix.shape
        np.testing.assert_allclose(matrix, expected_matrix)

    def test_find_optimal_speakers_assignment(self, speaker_assignment_pair: TranscriptPair) -> None:
        speaker_assignment_transcript_metrics = TranscriptMetrics(pair=speaker_assignment_pair)
        assignment = speaker_assignment_transcript_metrics._find_optimal_speakers_assignment()
        assert assignment['spk_can_1'] == 'spk_gold_1'
        assert assignment['spk_can_2'] == 'spk_gold_2'

    @pytest.fixture
    def candidate_from_source(self) -> Transcript:
        return Transcript(
            id='Test_transcript.mp3',
            language='en',
            segments=[
                Segment(id='2', text='A', speaker_id='spk_1',
                        timecodes=Timecodes(start_time=14.979, end_time=22.020),
                        segment_references={'2'}),
                Segment(id='3', text='B', speaker_id='spk_2',
                        timecodes=Timecodes(start_time=22.020, end_time=38.509),
                        segment_references={'3'}),
                Segment(id='4', text='C', speaker_id='spk_3',
                        timecodes=Timecodes(start_time=39.779, end_time=41.669),
                        segment_references={'4'}),
                Segment(id='5', text='D', speaker_id='spk_0',
                        timecodes=Timecodes(start_time=41.669, end_time=42.180),
                        segment_references={'5'})],
            # Keep a speakers list so _speaker_metrics can compute missing/extra counts
            speakers=[Speaker(speaker_id='spk_1', role='host'),
                      Speaker(speaker_id='spk_2', role='guest'),
                      Speaker(speaker_id='spk_3', role='guest'),
                      Speaker(speaker_id='spk_0', role='guest')]
        )

    def test_speaker_metrics(self,
                             transcript_metrics: TranscriptMetrics,
                             candidate_from_source: Transcript) -> None:
        # Using a candidate built inline from source (with self references)
        pair = TranscriptPair(gold=transcript_metrics.pair.gold,
                              candidate=candidate_from_source,
                              source=transcript_metrics.pair.source)
        source_as_candidate_metrics = TranscriptMetrics(pair=pair)
        result = source_as_candidate_metrics._speaker_metrics()

        expected_outcome = SpeakerMetrics(
            missing_count=0,
            extra_count=2,
            hits=2,
            substitutions=0,
            insertions=2,
            deletions=0,
        )

        assert result == expected_outcome

    def test_timing_metrics(self, transcript_metrics: TranscriptMetrics) -> None:
        result = transcript_metrics._timing_metrics()

        assert result == TimingMetrics(
            segments_compared=2,
            start_mean_abs_diff=0.0,
            end_mean_abs_diff=0.0,
            max_diff=0.0)
