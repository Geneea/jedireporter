import pytest

from jedireporter.eval.transcript_fix.models import (
    MetricsAggregator,
    MetricsCollector,
    SegmentationMetrics,
    SpeakerMetrics,
    TextMetrics,
    TextMetricsSegment,
    TextMetricsTranscript,
    TimingMetrics,
)


# -------------------- Transcript Metrics --------------------

@pytest.fixture
def text_transcript_metrics_1() -> TextMetricsTranscript:
    return TextMetricsTranscript(
        candidate2source_wer=0.2,
        candidate_word_count=100,
        source_word_count=110,
        candidate2gold_wer=0.22,
        source2gold_wer=0.25,
        gold_word_count=105,
    )


@pytest.fixture
def text_transcript_metrics_2() -> TextMetricsTranscript:
    return TextMetricsTranscript(
        candidate2source_wer=0.4,
        candidate_word_count=150,
        source_word_count=160,
        candidate2gold_wer=0.30,
        source2gold_wer=0.35,
        gold_word_count=155,
    )


def test_transcript_add_divide_delta(text_transcript_metrics_1: TextMetricsTranscript,
                                     text_transcript_metrics_2: TextMetricsTranscript) -> None:
    total = text_transcript_metrics_1 + text_transcript_metrics_2
    assert pytest.approx(total.candidate2source_wer) == 0.6
    assert total.candidate_word_count == 250
    assert total.source_word_count == 270
    assert total.gold_word_count == 260
    assert pytest.approx(total.candidate2gold_wer) == 0.52
    assert pytest.approx(total.source2gold_wer) == 0.60

    averaged = total.divide(nbr_candidates=2, nbr_gold=2)
    assert pytest.approx(averaged.candidate2source_wer) == 0.3
    assert pytest.approx(averaged.candidate_word_count) == 125
    assert pytest.approx(averaged.source_word_count) == 135
    assert pytest.approx(averaged.gold_word_count) == 130
    assert pytest.approx(averaged.candidate2gold_wer) == 0.26
    assert pytest.approx(averaged.source2gold_wer) == 0.30

    # candidate_source_delta = cand2gold - src2gold
    assert pytest.approx(averaged.candidate_source_delta) == 0.26 - 0.30


def test_transcript_zero_and_radd(text_transcript_metrics_1: TextMetricsTranscript) -> None:
    total = sum([text_transcript_metrics_1], start=TextMetrics.zero().transcript_level)
    assert total == text_transcript_metrics_1


# -------------------- Text Segment Metrics --------------------

@pytest.fixture
def text_segment_metrics_1() -> TextMetricsSegment:
    return TextMetricsSegment(
        candidate_segment_count=3,
        source_segment_count=4,
        omitted_source_segment_count=1,
        average_wer=0.2,
        max_wer=0.5,
        max_wer_segment_id='A: s2',
    )


@pytest.fixture
def text_segment_metrics_2() -> TextMetricsSegment:
    return TextMetricsSegment(
        candidate_segment_count=2,
        source_segment_count=2,
        omitted_source_segment_count=0,
        average_wer=0.4,
        max_wer=0.7,
        max_wer_segment_id='B: s9',
    )


def test_text_segment_add_and_divide_keeps_max(text_segment_metrics_1: TextMetricsSegment,
                                               text_segment_metrics_2: TextMetricsSegment) -> None:
    total = text_segment_metrics_1 + text_segment_metrics_2
    assert total.candidate_segment_count == 5
    assert total.source_segment_count == 6
    assert total.omitted_source_segment_count == 1
    assert pytest.approx(total.average_wer) == 0.6
    assert pytest.approx(total.max_wer) == 0.7
    assert total.max_wer_segment_id == 'B: s9'

    averaged = total.divide(nbr_candidates=5)
    assert pytest.approx(averaged.candidate_segment_count) == 1.0
    assert pytest.approx(averaged.source_segment_count) == 6 / 5
    assert pytest.approx(averaged.omitted_source_segment_count) == 0.2
    assert pytest.approx(averaged.average_wer) == 0.12
    assert averaged.max_wer == 0.7
    assert averaged.max_wer_segment_id == 'B: s9'


def test_text_segment_zero_and_radd(text_segment_metrics_1: TextMetricsSegment) -> None:
    total = sum([text_segment_metrics_1], start=TextMetrics.zero().segment_level)
    assert total == text_segment_metrics_1


# -------------------- Segmentation Metrics --------------------

@pytest.fixture
def segmentation_metrics_1() -> SegmentationMetrics:
    return SegmentationMetrics(
        precision=0.5,
        recall=0.25,
        f1=2 * 0.5 * 0.25 / (0.5 + 0.25),
        candidate_segment_count=4,
        gold_segment_count=8,
    )


@pytest.fixture
def segmentation_metrics_2() -> SegmentationMetrics:
    return SegmentationMetrics(
        precision=0.75,
        recall=0.5,
        f1=2 * 0.75 * 0.5 / (0.75 + 0.5),
        candidate_segment_count=6,
        gold_segment_count=12,
    )


def test_segmentation_add_and_div(segmentation_metrics_1: SegmentationMetrics,
                                  segmentation_metrics_2: SegmentationMetrics) -> None:
    total = segmentation_metrics_1 + segmentation_metrics_2
    assert pytest.approx(total.precision) == 1.25
    assert pytest.approx(total.recall) == 0.75
    assert pytest.approx(total.f1) == 14 / 15
    assert total.candidate_segment_count == 10
    assert total.gold_segment_count == 20

    avg = total / 2
    assert pytest.approx(avg.precision) == 0.625
    assert pytest.approx(avg.recall) == 0.375
    assert pytest.approx(avg.f1) == 7 / 15
    assert pytest.approx(avg.candidate_segment_count) == 5
    assert pytest.approx(avg.gold_segment_count) == 10


def test_segmentation_zero_and_radd(segmentation_metrics_1: SegmentationMetrics) -> None:
    total = sum([segmentation_metrics_1], start=SegmentationMetrics.zero())
    assert total == segmentation_metrics_1


# -------------------- Speaker Metrics --------------------

@pytest.fixture
def speaker_metrics_1() -> SpeakerMetrics:
    return SpeakerMetrics(missing_count=1, extra_count=0, hits=4, substitutions=2, insertions=1, deletions=0)


@pytest.fixture
def speaker_metrics_2() -> SpeakerMetrics:
    return SpeakerMetrics(missing_count=0, extra_count=2, hits=5, substitutions=1, insertions=0, deletions=3)


def test_speaker_add_and_div(speaker_metrics_1: SpeakerMetrics,
                             speaker_metrics_2: SpeakerMetrics) -> None:
    total = speaker_metrics_1 + speaker_metrics_2
    assert total == SpeakerMetrics(missing_count=1, extra_count=2, hits=9, substitutions=3, insertions=1, deletions=3)

    avg = total / 2
    assert pytest.approx(avg.missing_count) == 0.5
    assert pytest.approx(avg.extra_count) == 1.0
    assert pytest.approx(avg.hits) == 4.5
    assert pytest.approx(avg.substitutions) == 1.5
    assert pytest.approx(avg.insertions) == 0.5
    assert pytest.approx(avg.deletions) == 1.5


def test_speaker_zero_and_radd(speaker_metrics_1: SpeakerMetrics) -> None:
    total = sum([speaker_metrics_1], start=SpeakerMetrics.zero())
    assert total == speaker_metrics_1


# -------------------- Timing Metrics --------------------

@pytest.fixture
def timing_metrics_1() -> TimingMetrics:
    return TimingMetrics(segments_compared=4, start_mean_abs_diff=0.1, end_mean_abs_diff=0.2, max_diff=0.6)


@pytest.fixture
def timing_metrics_2() -> TimingMetrics:
    return TimingMetrics(segments_compared=6, start_mean_abs_diff=0.2, end_mean_abs_diff=0.3, max_diff=0.4)


def test_timing_add_and_div_keeps_max(timing_metrics_1: TimingMetrics,
                                      timing_metrics_2: TimingMetrics) -> None:
    total = timing_metrics_1 + timing_metrics_2
    assert total.segments_compared == 10
    assert pytest.approx(total.start_mean_abs_diff) == 0.3
    assert pytest.approx(total.end_mean_abs_diff) == 0.5
    assert total.max_diff == 0.6

    avg = total / 2
    assert pytest.approx(avg.segments_compared) == 5
    assert pytest.approx(avg.start_mean_abs_diff) == 0.15
    assert pytest.approx(avg.end_mean_abs_diff) == 0.25
    assert avg.max_diff == 0.6


def test_timing_zero_and_radd(timing_metrics_1: TimingMetrics) -> None:
    total = sum([timing_metrics_1], start=TimingMetrics.zero())
    assert total == timing_metrics_1


def test_timing_zero() -> None:
    zero = TimingMetrics.zero()
    assert zero == TimingMetrics(segments_compared=0.0, start_mean_abs_diff=0.0, end_mean_abs_diff=0.0, max_diff=0.0)


# -------------------- Metrics Aggregator --------------------
@pytest.fixture
def text_transcript_metrics_no_gold() -> TextMetricsTranscript:
    return TextMetricsTranscript(
        candidate2source_wer=0.4,
        candidate_word_count=150,
        source_word_count=160,
        candidate2gold_wer=None,
        source2gold_wer=None,
        gold_word_count=None,
    )


@pytest.fixture
def text_segment_metrics_no_gold() -> TextMetricsSegment:
    return TextMetricsSegment(
        candidate_segment_count=2,
        source_segment_count=3,
        omitted_source_segment_count=1,
        average_wer=0.2,
        max_wer=0.25,
        max_wer_segment_id='B: s1',
    )


def test_metrics_aggregator_summary_mixed_samples(
        text_transcript_metrics_1: TextMetricsTranscript,
        text_transcript_metrics_no_gold: TextMetricsTranscript,
        text_segment_metrics_1: TextMetricsSegment,
        text_segment_metrics_no_gold: TextMetricsSegment,
        segmentation_metrics_1: SegmentationMetrics,
        speaker_metrics_1: SpeakerMetrics,
        timing_metrics_1: TimingMetrics,
        timing_metrics_2: TimingMetrics,
) -> None:
    # one sample with gold-dependent metrics, one without (no segmentation)
    text_1 = TextMetrics(transcript_level=text_transcript_metrics_1, segment_level=text_segment_metrics_1)
    text_2 = TextMetrics(transcript_level=text_transcript_metrics_no_gold, segment_level=text_segment_metrics_no_gold)

    metrics_1 = MetricsCollector(id='A', text_metrics=text_1, segmentation_metrics=segmentation_metrics_1,
                                 speaker_metrics=speaker_metrics_1, timing_metrics=timing_metrics_1)
    metrics_2 = MetricsCollector(id='B', text_metrics=text_2, segmentation_metrics=None,
                                 speaker_metrics=None, timing_metrics=timing_metrics_2)

    aggregator = MetricsAggregator()
    aggregator.add(metrics_1)
    aggregator.add(metrics_2)
    summary = aggregator.summary

    assert summary.id == 'summary'

    # Text: candidate2source averaged over 2 candidates; gold-based divided by gold_count = 1
    assert pytest.approx(summary.text_metrics.transcript_level.candidate2source_wer) == (0.2 + 0.4) / 2
    assert pytest.approx(summary.text_metrics.transcript_level.candidate2gold_wer) == 0.22
    assert pytest.approx(summary.text_metrics.transcript_level.source2gold_wer) == 0.25
    assert pytest.approx(summary.text_metrics.transcript_level.candidate_word_count) == (100 + 150) / 2

    # Segmentation/Speaker: averaged only over gold_count = 1
    assert summary.segmentation_metrics == segmentation_metrics_1
    assert summary.speaker_metrics == speaker_metrics_1

    # Timing: averaged over both candidates
    assert pytest.approx(summary.timing_metrics.segments_compared) == 5
    assert pytest.approx(summary.timing_metrics.max_diff) == 0.6
