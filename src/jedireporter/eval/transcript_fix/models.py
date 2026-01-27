"""
Metrics aggregation models used by transcript evaluation.

Note on addition vs. averaging
------------------------------
For many fields in these models (e.g., WER averages, precision/recall/f1,
mean absolute timing differences), the ``__add__`` implementations deliberately
accumulate raw sums rather than returning a meaningful, normalized metric.
These intermediate sums can look nonsensical if interpreted as final values.

This is intentional to make it possible to:
- use Python's ``sum(...)`` with a neutral ``zero()`` value, and then
- normalize exactly once via ``divide(...)`` or ``/__truediv__`` in the caller.

Exceptions are 'max' fields (e.g., ``max_wer``, ``max_diff``), which are kept
as the maximum when adding.

In short: add to accumulate; divide to interpret. Do not read intermediate sums
as final metrics.
"""
from pydantic import computed_field, Field

from jedireporter.camelModel import CamelModel, FrozenCamelModel
from jedireporter.eval.transcript_fix.utils import safe_sum


class TextMetricsTranscript(FrozenCamelModel):
    candidate2source_wer: float
    # Counts are ints per-sample; allow float after averaging
    candidate_word_count: int | float
    source_word_count: int | float
    gold_word_count: int | float | None = None
    candidate2gold_wer: float | None = None
    source2gold_wer: float | None = None

    @computed_field
    @property
    def candidate_source_delta(self) -> float | None:
        if self.candidate2gold_wer is not None and self.source2gold_wer is not None:
            return self.candidate2gold_wer - self.source2gold_wer
        else:
            return None

    def __add__(self, other: 'TextMetricsTranscript') -> 'TextMetricsTranscript':
        if not isinstance(other, TextMetricsTranscript):
            raise ValueError(f'Cannot add: {other}')
        # Note: This returns accumulated sums. Call 'divide(...)' later to obtain
        # per-sample averages. Optional gold-related fields are combined via 'safe_sum'.
        return TextMetricsTranscript(
            candidate2source_wer=self.candidate2source_wer + other.candidate2source_wer,
            candidate_word_count=self.candidate_word_count + other.candidate_word_count,
            source_word_count=self.source_word_count + other.source_word_count,
            candidate2gold_wer=safe_sum(self.candidate2gold_wer, other.candidate2gold_wer),
            source2gold_wer=safe_sum(self.source2gold_wer, other.source2gold_wer),
            gold_word_count=safe_sum(self.gold_word_count, other.gold_word_count), )

    def divide(self, nbr_candidates: int, nbr_gold: int) -> 'TextMetricsTranscript':
        if nbr_candidates <= 0:
            raise ValueError('nbr_candidate must be positive integer')
        # Aggregator ensures 'nbr_gold' equals the number of samples that actually contribute gold-dependent fields;
        # this prevents division by zero for gold-only metrics. Therefore, this error should never be raised.
        if nbr_gold < 0 or nbr_gold == 0 and any((self.candidate2gold_wer, self.source2gold_wer, self.gold_word_count)):
            raise ValueError('nbr_gold must be positive integer or gold-related metrics must be None (no gold samples)')

        return TextMetricsTranscript(
            candidate2source_wer=self.candidate2source_wer / nbr_candidates,
            candidate_word_count=self.candidate_word_count / nbr_candidates,
            source_word_count=self.source_word_count / nbr_candidates,
            candidate2gold_wer=None if self.candidate2gold_wer is None else self.candidate2gold_wer / nbr_gold,
            source2gold_wer=None if self.source2gold_wer is None else self.source2gold_wer / nbr_gold,
            gold_word_count=None if self.gold_word_count is None else self.gold_word_count / nbr_gold,
        )

    @classmethod
    def zero(cls) -> 'TextMetricsTranscript':
        return TextMetricsTranscript(
            candidate2source_wer=0.0,
            candidate_word_count=0.0,
            source_word_count=0.0,
            # Gold-dependent fields are intentionally None (not 0.0) so that
            # non-gold samples do not contribute artificial zeros during accumulation.
            candidate2gold_wer=None,
            source2gold_wer=None,
            gold_word_count=None,
        )


class TextMetricsSegment(FrozenCamelModel):
    # Counts are ints per-sample; allow float after averaging
    candidate_segment_count: int | float
    source_segment_count: int | float
    omitted_source_segment_count: int | float
    average_wer: float | None
    max_wer: float | None
    max_wer_segment_id: str | None

    @staticmethod
    def _safe_greater_than(a: float | None, b: float | None) -> float | None:
        """Returns `None` if both values are `None`, otherwise returns non-None element or the greater one"""
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return a if a > b else b

    def __add__(self, other: 'TextMetricsSegment') -> 'TextMetricsSegment':
        """Max field is not added but the maximum of the two values is returned with respective `max_wer_segment_id`"""
        if not isinstance(other, TextMetricsSegment):
            raise ValueError(f'Cannot add: {other}')

        max_wer = self._safe_greater_than(self.max_wer, other.max_wer)
        max_wer_segment_id = None
        if max_wer == self.max_wer:
            max_wer_segment_id = self.max_wer_segment_id
        elif max_wer == other.max_wer:
            max_wer_segment_id = other.max_wer_segment_id
        # Note: counts and average_wer are accumulated here to be normalized later
        # via 'divide(nbr_candidates=...)'. 'max_wer' and its id are preserved as a max.
        return TextMetricsSegment(
            candidate_segment_count=self.candidate_segment_count + other.candidate_segment_count,
            source_segment_count=self.source_segment_count + other.source_segment_count,
            omitted_source_segment_count=self.omitted_source_segment_count + other.omitted_source_segment_count,
            average_wer=safe_sum(self.average_wer, other.average_wer),
            max_wer=max_wer,
            max_wer_segment_id=max_wer_segment_id,
        )

    def divide(self, nbr_candidates: int) -> 'TextMetricsSegment':
        """Max field is kept and not divided."""
        if nbr_candidates <= 0:
            raise ValueError('nbr_candidate must be positive integer')

        return TextMetricsSegment(
            candidate_segment_count=self.candidate_segment_count / nbr_candidates,
            source_segment_count=self.source_segment_count / nbr_candidates,
            omitted_source_segment_count=self.omitted_source_segment_count / nbr_candidates,
            average_wer=None if self.average_wer is None else self.average_wer / nbr_candidates,
            max_wer=self.max_wer,
            max_wer_segment_id=self.max_wer_segment_id,
        )

    @classmethod
    def zero(cls):
        return TextMetricsSegment(
            candidate_segment_count=0.0,
            source_segment_count=0.0,
            omitted_source_segment_count=0.0,
            average_wer=None,
            max_wer=None,
            max_wer_segment_id=None,
        )


class TextMetrics(FrozenCamelModel):
    transcript_level: TextMetricsTranscript
    segment_level: TextMetricsSegment

    def __add__(self, other: 'TextMetrics') -> 'TextMetrics':
        if not isinstance(other, TextMetrics):
            raise ValueError(f'Cannot add: {other}')
        # Component-wise accumulation; call 'divide(...)' later to obtain averages.
        return TextMetrics(
            transcript_level=self.transcript_level + other.transcript_level,
            segment_level=self.segment_level + other.segment_level,
        )

    def divide(self, nbr_candidates: int, nbr_gold: int) -> 'TextMetrics':
        if nbr_candidates <= 0:
            raise ValueError('nbr_candidate must be positive integer')
        return TextMetrics(
            transcript_level=self.transcript_level.divide(nbr_candidates=nbr_candidates, nbr_gold=nbr_gold),
            segment_level=self.segment_level.divide(nbr_candidates=nbr_candidates),
        )

    @classmethod
    def zero(cls) -> 'TextMetrics':
        return cls(
            transcript_level=TextMetricsTranscript.zero(),
            segment_level=TextMetricsSegment.zero()
        )


class SegmentationMetrics(FrozenCamelModel):
    precision: float
    recall: float
    f1: float
    # Counts are ints per-sample; allow float after averaging
    candidate_segment_count: int | float
    gold_segment_count: int | float

    def __add__(self, other: 'SegmentationMetrics') -> 'SegmentationMetrics':
        if not isinstance(other, SegmentationMetrics):
            raise ValueError(f'Cannot add: {other}')
        # Accumulate sums to allow macro-averaging later via '/__truediv__'.
        return SegmentationMetrics(
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1=self.f1 + other.f1,
            candidate_segment_count=self.candidate_segment_count + other.candidate_segment_count,
            gold_segment_count=self.gold_segment_count + other.gold_segment_count,
        )

    def __truediv__(self, n: int) -> 'SegmentationMetrics':
        if not isinstance(n, int):
            raise ValueError(f'Cannot divide by {n}')
        if n <= 0:
            raise ValueError('nbr_candidate must be positive integer')
        return SegmentationMetrics(
            precision=self.precision / n,
            recall=self.recall / n,
            f1=self.f1 / n,
            candidate_segment_count=self.candidate_segment_count / n,
            gold_segment_count=self.gold_segment_count / n,
        )

    @classmethod
    def zero(cls) -> 'SegmentationMetrics':
        return SegmentationMetrics(
            precision=0,
            recall=0,
            f1=0,
            candidate_segment_count=0,
            gold_segment_count=0
        )


class SpeakerMetrics(FrozenCamelModel):
    # Counts are ints per-sample; allow float after averaging
    missing_count: int | float
    extra_count: int | float
    hits: int | float = Field(
        description='The number of correct speakers between gold and candidate segments')
    substitutions: int | float = Field(
        description='The number of substitutions required to transform gold segment speakers to candidate ones')
    insertions: int | float = Field(
        description='The number of insertions required to transform gold segment speakers to candidate ones')
    deletions: int | float = Field(
        description='The number of deletions required to transform gold segment speakers to candidate ones')

    def __add__(self, other: 'SpeakerMetrics') -> 'SpeakerMetrics':
        if not isinstance(other, SpeakerMetrics):
            raise ValueError(f'Cannot add: {other}')
        # Accumulate sums; normalize later with '/__truediv__' when averaging.
        return SpeakerMetrics(
            missing_count=self.missing_count + other.missing_count,
            extra_count=self.extra_count + other.extra_count,
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions)

    def __truediv__(self, n: int) -> 'SpeakerMetrics':
        if not isinstance(n, int):
            raise ValueError(f'Cannot divide by {n}')
        if n <= 0:
            raise ValueError('n must be positive integer')
        return SpeakerMetrics(
            missing_count=self.missing_count / n,
            extra_count=self.extra_count / n,
            hits=self.hits / n,
            substitutions=self.substitutions / n,
            insertions=self.insertions / n,
            deletions=self.deletions / n)

    @classmethod
    def zero(cls) -> 'SpeakerMetrics':
        return SpeakerMetrics(
            missing_count=0,
            extra_count=0,
            hits=0,
            substitutions=0,
            insertions=0,
            deletions=0
        )


class TimingMetrics(FrozenCamelModel):
    segments_compared: int | float
    start_mean_abs_diff: float
    end_mean_abs_diff: float
    max_diff: float

    def __add__(self, other: 'TimingMetrics') -> 'TimingMetrics':
        """Max field is not added but the maximum of the two values is returned."""
        if not isinstance(other, TimingMetrics):
            raise ValueError(f'Cannot add: {other}')
        # Sum the mean-like fields; keep 'max_diff' as the maximum.
        return TimingMetrics(
            segments_compared=self.segments_compared + other.segments_compared,
            start_mean_abs_diff=self.start_mean_abs_diff + other.start_mean_abs_diff,
            end_mean_abs_diff=self.end_mean_abs_diff + other.end_mean_abs_diff,
            max_diff=max(self.max_diff, other.max_diff))

    def __truediv__(self, n: int) -> 'TimingMetrics':
        """Max field is kept and not divided."""
        if not isinstance(n, int):
            raise ValueError(f'Cannot divide by {n}')
        if n <= 0:
            raise ValueError('n must be positive integer')
        return TimingMetrics(
            segments_compared=self.segments_compared / n,
            start_mean_abs_diff=self.start_mean_abs_diff / n,
            end_mean_abs_diff=self.end_mean_abs_diff / n,
            max_diff=self.max_diff)

    @classmethod
    def zero(cls) -> 'TimingMetrics':
        return cls(
            segments_compared=0.0,
            start_mean_abs_diff=0.0,
            end_mean_abs_diff=0.0,
            max_diff=0.0,
        )


class MetricsCollector(FrozenCamelModel):
    id: str
    text_metrics: TextMetrics
    segmentation_metrics: SegmentationMetrics | None = None
    speaker_metrics: SpeakerMetrics | None = None
    timing_metrics: TimingMetrics | None


class MetricsAggregator(CamelModel):
    per_sample: list[MetricsCollector] = Field(default_factory=list)

    def add(self, metrics: MetricsCollector) -> None:
        self.per_sample.append(metrics)

    @computed_field
    @property
    def summary(self) -> MetricsCollector | None:
        if not self.per_sample:
            return None

        candidates_count = len(self.per_sample)
        # Using presence of 'segmentation_metrics' as a proxy for samples that have gold
        # (gold-dependent metrics). Only those are used for gold-based averaging ('gold_count').
        segments = [c.segmentation_metrics for c in self.per_sample if c.segmentation_metrics is not None]
        speakers = [c.speaker_metrics for c in self.per_sample if c.speaker_metrics is not None]
        gold_count = len(segments)
        # Pattern: accumulate with 'sum(...)' using a neutral zero, then normalize once
        # via 'divide(...)' or '/__truediv__'. This avoids repeated rounding and keeps
        # logic uniform for candidate-only vs gold-dependent metrics.
        text_total = sum((collector.text_metrics for collector in self.per_sample), start=TextMetrics.zero())
        text_metrics = text_total.divide(nbr_candidates=candidates_count, nbr_gold=gold_count)

        segmentation_total = sum(segments, start=SegmentationMetrics.zero()) if segments else None
        segmentation_metrics = (segmentation_total / gold_count) if segmentation_total is not None else None

        speaker_total = sum(speakers, start=SpeakerMetrics.zero()) if speakers else None
        speaker_metrics = (speaker_total / gold_count) if speaker_total is not None else None

        timings = [c.timing_metrics for c in self.per_sample if c.timing_metrics is not None]
        if timings_len := len(timings):
            timing_metrics = sum(timings, start=TimingMetrics.zero()) / timings_len
        else:
            timing_metrics = None

        return MetricsCollector(
            id='summary',
            text_metrics=text_metrics,
            segmentation_metrics=segmentation_metrics,
            speaker_metrics=speaker_metrics,
            timing_metrics=timing_metrics)
