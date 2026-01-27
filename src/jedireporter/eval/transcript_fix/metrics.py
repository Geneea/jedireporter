from collections import defaultdict
from statistics import mean

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

from jedireporter.eval.transcript_fix.models import (
    MetricsCollector,
    SegmentationMetrics,
    SpeakerMetrics,
    TextMetrics,
    TextMetricsSegment,
    TextMetricsTranscript,
    TimingMetrics,
)
from jedireporter.eval.transcript_fix.utils import TranscriptPair
from jedireporter.transcript import Segment, Transcript
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)


class TranscriptMetrics:
    def __init__(self, pair: TranscriptPair) -> None:
        self.pair = pair

    def compute(self) -> MetricsCollector:
        metrics = MetricsCollector(
            id=self.pair.source.id,
            text_metrics=self._text_metrics(),
            segmentation_metrics=self._segmentation_metrics(),
            speaker_metrics=self._speaker_metrics(),
            timing_metrics=self._timing_metrics())
        return metrics

    @staticmethod
    def _join_transcript(transcript: Transcript) -> str:
        return ' '.join(segment.text.strip() for segment in transcript.segments if segment.text)

    def _text_metrics_transcript_level(self) -> TextMetricsTranscript:
        candidate_text = self._join_transcript(self.pair.candidate)
        source_text = self._join_transcript(self.pair.source)
        candidate_word_count = len(candidate_text.split())
        source_word_count = len(source_text.split())
        candidate2source_wer = jiwer.wer(source_text, candidate_text)
        if self.pair.gold:
            gold_text = self._join_transcript(self.pair.gold)
            return TextMetricsTranscript(
                candidate2source_wer=candidate2source_wer,
                candidate_word_count=candidate_word_count,
                source_word_count=source_word_count,
                candidate2gold_wer=jiwer.wer(gold_text, candidate_text),
                source2gold_wer=jiwer.wer(gold_text, source_text),
                gold_word_count=len(gold_text.split())
            )
        else:
            return TextMetricsTranscript(
                candidate2source_wer=candidate2source_wer,
                candidate_word_count=candidate_word_count,
                source_word_count=source_word_count)

    @staticmethod
    def _concatenate_source_segments(candidate_segment: Segment,
                                     source_order: dict[str, int],
                                     source_by_id: dict[str, Segment],
                                     candidate_transcript: Transcript) -> tuple[str, set[str]]:
        """
        Sorts referenced source segments and concatenates their text.

        :param candidate_segment: candidate segment with referenced source segments
        :param source_order: dictionary mapping source segment IDs to its sorted position in original transcript
        :param source_by_id: dictionary mapping source segment IDs to respective ``Segment`` objects
        :param candidate_transcript: candidate transcript with referenced source segments
        :return: tuple of concatenated source segments text and set of their IDs
        """
        used_source_ids: set[str] = set()
        source_references = sorted(
            candidate_segment.segment_references,
            key=lambda _id: source_order.get(_id, float('inf')),
        )
        source_text = ''
        for source_reference in source_references:
            if source_reference not in source_by_id:
                LOG.warning(f'[{candidate_transcript.id}] Candidate segment {candidate_segment.id} references '
                            f'source segment with id {source_reference} which is not in source segments')
                continue
            source_text += ' ' + source_by_id[source_reference].text.strip()
            used_source_ids.add(source_reference)
        return source_text, used_source_ids

    def _text_metrics_segment_level(self) -> TextMetricsSegment:
        source_transcript = self.pair.source
        candidate_transcript = self.pair.candidate

        source_by_id: dict[str, Segment] = {segment.id: segment for segment in source_transcript.segments}
        source_order: dict[str, int] = {segment.id: idx for idx, segment in enumerate(source_transcript.segments)}

        wer_per_segment: list[tuple[str, float]] = []
        referenced_source_ids: set[str] = set()

        for candidate_segment in candidate_transcript.segments:
            candidate_text = candidate_segment.text.strip()
            source_text, used_source_ids = self._concatenate_source_segments(candidate_segment, source_order,
                                                                             source_by_id, candidate_transcript)
            referenced_source_ids |= used_source_ids
            wer_value = jiwer.wer(source_text, candidate_text)
            wer_per_segment.append((candidate_segment.id, wer_value))

        omitted_source_ids = source_by_id.keys() - referenced_source_ids
        average_wer = mean([value for _, value in wer_per_segment]) if wer_per_segment else None
        max_pair = max(wer_per_segment, key=lambda pair: pair[1]) if wer_per_segment else None

        return TextMetricsSegment(
            candidate_segment_count=len(candidate_transcript.segments),
            source_segment_count=len(source_transcript.segments),
            omitted_source_segment_count=len(omitted_source_ids),
            average_wer=average_wer,
            max_wer=max_pair[1] if max_pair else None,
            # Added also transcript id for easier recognition in results summary
            max_wer_segment_id=f'{source_transcript.id}: {max_pair[0]}' if max_pair else None,
        )

    def _text_metrics(self) -> TextMetrics:
        transcript_level_metrics = self._text_metrics_transcript_level()
        segment_level_metrics = self._text_metrics_segment_level()
        return TextMetrics(transcript_level=transcript_level_metrics, segment_level=segment_level_metrics)

    @staticmethod
    def _get_segment_references(transcript: Transcript) -> set[frozenset[str]]:
        segment_references: set[frozenset[str]] = {frozenset(segment.segment_references) for segment in
                                                   transcript.segments}
        return segment_references

    def _segmentation_metrics(self) -> SegmentationMetrics | None:
        """
        Counts the number of correctly split segments by comparing ``segment_references`` of the candidate and gold
        ``Transcript``.
        """
        if self.pair.gold is None:
            return None

        candidate_references = self._get_segment_references(self.pair.candidate)
        gold_references = self._get_segment_references(self.pair.gold)
        if not candidate_references or not gold_references:
            raise ValueError('Candidate or gold transcript is missing references to original source segments.')
        true_positive = len(candidate_references & gold_references)
        recall = true_positive / len(gold_references)
        precision = true_positive / len(candidate_references)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return SegmentationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            candidate_segment_count=len(candidate_references),
            gold_segment_count=len(gold_references))

    @staticmethod
    def _overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    @staticmethod
    def _build_speaker_overlap_matrix_no_timecodes(
            gold: Transcript,
            candidate: Transcript,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Build a gold×candidate speaker overlap matrix measured in sizes of original (source transcript) segments
        overlap.

        :param gold: gold transcript.
        :param candidate: candidate transcript.
        :returns: (overlap_matrix, gold_speaker_ids, candidate_speaker_ids) where
                  ``overlap_matrix`` is square matrix with size ``max(len(gold_speaker_ids),
                  len(candidate_speaker_ids))`` and entry ``[i,j]`` is the total number of original segments, which are
                   assigned to both ``gold_speaker_ids[i]`` and ``candidate_speaker_ids[j]``.
        """
        gold_ids = [speaker.speaker_id for speaker in gold.speakers]
        candidate_ids = [speaker.speaker_id for speaker in candidate.speakers]
        gi = {speaker_id: i for i, speaker_id in enumerate(gold_ids)}
        cj = {speaker_id: j for j, speaker_id in enumerate(candidate_ids)}

        # square matrix (pad with zeros)
        n = max(len(gold_ids), len(candidate_ids))
        matrix = np.zeros((n, n), dtype=float)

        # Assign each speaker from both transcripts (gold and candidate), set of original segment IDs (from the source
        # transcript)
        candidate_to_original_segments = defaultdict(set)
        for segment in candidate.segments:
            candidate_to_original_segments[segment.speaker_id].add(frozenset(segment.segment_references))
        gold_to_original_segments = defaultdict(set)
        for segment in gold.segments:
            gold_to_original_segments[segment.speaker_id].add(frozenset(segment.segment_references))

        # Use intersection sizes as a proxy for speakers overlap
        for candidate_speaker in candidate_ids:
            for gold_speaker in gold_ids:
                g_idx = gi[gold_speaker]
                c_idx = cj[candidate_speaker]
                matrix[g_idx, c_idx] = len(
                    candidate_to_original_segments[candidate_speaker] & gold_to_original_segments[gold_speaker])
        return matrix, gold_ids, candidate_ids

    def _build_speaker_overlap_matrix(
            self,
            gold: Transcript,
            candidate: Transcript,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Build a gold×candidate speaker overlap matrix measured in seconds.

        :param gold: gold transcript.
        :param candidate: candidate transcript.
        :returns: (overlap_matrix, gold_speaker_ids, candidate_speaker_ids) where
                  ``overlap_matrix`` is square matrix with size ``max(len(gold_speaker_ids),
                  len(candidate_speaker_ids))`` and entry ``[i,j]`` is the total temporal overlap (seconds) between
                  ``gold_speaker_ids[i]`` and ``candidate_speaker_ids[j]``.
        """
        gold_ids = sorted({s.speaker_id for s in gold.segments if s.timecodes is not None})
        candidate_ids = sorted({s.speaker_id for s in candidate.segments if s.timecodes is not None})
        if not gold_ids or not candidate_ids:
            return np.zeros((0, 0), dtype=float), gold_ids, candidate_ids

        gi = {speaker_id: i for i, speaker_id in enumerate(gold_ids)}
        cj = {speaker_id: j for j, speaker_id in enumerate(candidate_ids)}

        # square matrix (pad with zeros)
        n = max(len(gold_ids), len(candidate_ids))
        matrix = np.zeros((n, n), dtype=float)
        # segments sorted by start_time
        g_segments = sorted([s for s in gold.segments if s.timecodes is not None],
                            key=lambda s: s.timecodes.start_time)
        c_segments = sorted([s for s in candidate.segments if s.timecodes is not None],
                            key=lambda s: s.timecodes.start_time)

        j = 0
        for gs in g_segments:
            g_idx = gi[gs.speaker_id]
            gs_start, gs_end = gs.timecodes.start_time, gs.timecodes.end_time
            # candidate end_time must be > gold start_time
            while j < len(c_segments) and c_segments[j].timecodes.end_time <= gs_start:
                j += 1
            k = j
            # check if current candidate start_time is before gold end_time and if so, compute the overlap and add it
            # to the matrix
            while k < len(c_segments) and c_segments[k].timecodes.start_time < gs_end:
                cs = c_segments[k]
                c_idx = cj[cs.speaker_id]
                cs_start, cs_end = cs.timecodes.start_time, cs.timecodes.end_time
                matrix[g_idx, c_idx] += self._overlap_seconds(gs_start, gs_end, cs_start, cs_end)
                k += 1

        return matrix, gold_ids, candidate_ids

    def _find_optimal_speakers_assignment(self) -> dict[str, str]:
        """
        Returns a mapping dict: {candidate_speaker_id -> gold_speaker_id}
        that maximizes total time-overlap between mapped speakers.
        Unmapped speakers are omitted (treated as extras/missing).
        """
        if self.pair.gold is None:
            return {}

        gold = self.pair.gold
        candidate = self.pair.candidate

        if not any((segment.timecodes for segment in gold.segments)):
            LOG.warning(f'[{gold.id}] No timecodes found in gold segment, using segment references for building '
                        f'approximate overlap matrix instead!')
            matrix, gold_ids, candidate_ids = self._build_speaker_overlap_matrix_no_timecodes(gold, candidate)
        else:
            matrix, gold_ids, candidate_ids = self._build_speaker_overlap_matrix(gold, candidate)
        if matrix.size == 0:
            return {}

        rows, cols = linear_sum_assignment(matrix, maximize=True)

        mapping: dict[str, str] = {}
        for r, c in zip(rows, cols):
            # only keep real speaker indices (skip padded rows/cols) with time overlap > 0
            if r < len(gold_ids) and c < len(candidate_ids) and matrix[r, c] > 0.0:
                mapping[candidate_ids[c]] = gold_ids[r]

        return mapping

    @staticmethod
    def _compute_speaker_statistics(candidate_segment_speakers: list[str],
                                    gold_segment_speakers: list[str]) -> tuple[int, int, int, int]:
        """
        Compute (hits, substitutions, insertions, deletions) by aligning two lists of speaker IDs with
        ``jiwer.process_words`` (gold = reference, candidate = hypothesis). This handles case where there is, for
        example, incorrect merge of segments and simple element-wise comparison would carry on the error from shift.


        :param candidate_segment_speakers: candidate speaker ID list (hypothesis).
        :param gold_segment_speakers: gold speaker ID list (reference).
        :return: tuple (hits, substitutions, insertions, deletions).
        """
        candidate_speakers_str = ' '.join((speaker.replace(' ', '_') for speaker in candidate_segment_speakers))
        gold_speakers_str = ' '.join((speaker.replace(' ', '_') for speaker in gold_segment_speakers))
        stats = jiwer.process_words(gold_speakers_str, candidate_speakers_str)

        return stats.hits, stats.substitutions, stats.insertions, stats.deletions

    def _speaker_metrics(self) -> SpeakerMetrics | None:
        """
        Compute speaker-assignment accuracy with **permutation-invariant** IDs.

        We first map candidate speaker IDs to gold IDs via an optimal assignment
        (to fix arbitrary label switching), then align the speaker sequences with
        ``jiwer`` to stay robust to segment merges/splits (counting them as
        insertions/deletions rather than shifting all errors).

        :returns: Per-sample speaker metrics (hits, substitutions, insertions, deletions)
                  plus counts of missing/extra speakers; ``None`` if no gold transcript.
        """
        if self.pair.gold is None:
            return None

        candidate = self.pair.candidate
        gold = self.pair.gold

        candidate_speakers_mapping = self._find_optimal_speakers_assignment()
        candidate_segment_speakers = [
            candidate_speakers_mapping.get(segment.speaker_id, f'candidate_{segment.speaker_id}')
            for segment in candidate.segments
        ]
        gold_segment_speakers = [segment.speaker_id for segment in gold.segments]
        hits, substitutions, insertions, deletions = self._compute_speaker_statistics(candidate_segment_speakers,
                                                                                      gold_segment_speakers)
        return SpeakerMetrics(
            missing_count=max(0, len(self.pair.gold.speakers) - len(self.pair.candidate.speakers)),
            extra_count=max(0, len(self.pair.candidate.speakers) - len(self.pair.gold.speakers)),
            hits=hits,
            substitutions=substitutions,
            insertions=insertions,
            deletions=deletions
        )

    def _timing_metrics(self) -> TimingMetrics | None:
        source = self.pair.source
        # Check if the segments contain timecodes
        if not all((segment.timecodes for segment in source.segments)):
            return None

        candidate = self.pair.candidate
        source_by_id: dict[str, Segment] = {segment.id: segment for segment in source.segments}
        source_order: dict[str, int] = {segment.id: idx for idx, segment in enumerate(source.segments)}

        start_diffs: list[float] = []
        end_diffs: list[float] = []
        for segment in candidate.segments:
            if segment.segment_references:
                refs_in_source = [ref_id for ref_id in segment.segment_references if ref_id in source_by_id]
                if not refs_in_source:
                    raise ValueError(
                        f'All referenced segments in candidate segment {segment.id} are missing from source segments.'
                    )
                start_id = min(refs_in_source, key=lambda _id: source_order[_id])
                end_id = max(refs_in_source, key=lambda _id: source_order[_id])
                start_diffs.append(abs(source_by_id[start_id].timecodes.start_time - segment.timecodes.start_time))
                end_diffs.append(abs(source_by_id[end_id].timecodes.end_time - segment.timecodes.end_time))
            else:
                raise ValueError(f'Missing segment references for candidate segment {segment.id}.')

        diffs = start_diffs + end_diffs
        max_diff_val = max(diffs) if diffs else 0.0
        return TimingMetrics(
            segments_compared=len(start_diffs),
            start_mean_abs_diff=mean(start_diffs) if start_diffs else 0.0,
            end_mean_abs_diff=mean(end_diffs) if end_diffs else 0.0,
            max_diff=max_diff_val)
