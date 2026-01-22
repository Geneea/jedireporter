from importlib.resources import read_text
from typing import Literal

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from jedireporter.llm import InstructorLLM
from jedireporter.transcript import (
    CorrectedText,
    CorrectedTextList,
    RelevantSegmentIds,
    Segment,
    SegmentBoundaryCorrection,
    SegmentBoundaryCorrectionList,
    SegmentSource,
    SegmentSplit,
    SegmentSplitList,
    Speaker,
    SpeakerList,
    Timecodes,
    Transcript
)
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)


class FixedState(BaseModel):
    source: Transcript  # the unmodified input
    merged_segments: list[Segment] | None = None  # source with neighboring same-speaker segments merged
    segments_with_fixed_boundaries_and_speakers: list[SegmentSource] | None = None
    relevant_segments_status: tuple[int, str] | None = None  # tuple of (attempt number, last error message)
    relevant_segments: list[Segment] | None = None
    detected_speakers: list[Speaker] | None = None
    split_segments: list[Segment] | None = None
    fixed_segments: list[Segment] | None = None
    fixed: Transcript | None = None  # input with fixed errors


class FixTranscriptNodes:
    def __init__(self, llm: InstructorLLM):
        self.llm: InstructorLLM = llm

    def build_subgraph(self) -> CompiledStateGraph:
        builder = StateGraph(FixedState)
        builder.add_node('merge_same_speaker_neighbors', self.merge_same_speaker_neighbors)
        builder.add_node('detect_speakers', self.detect_speakers)
        builder.add_node('fix_segment_boundaries_and_speakers', self.fix_segment_boundaries_and_speakers)
        builder.add_node('extract_relevant_segments', self.extract_relevant_segments)
        builder.add_node('split_segments', self.split_segments)
        builder.add_node('fix_grammar', self.fix_grammar)
        builder.add_node('construct_fixed_transcript', self.construct_fixed_transcript)

        builder.add_edge(START, 'fix_segment_boundaries_and_speakers')
        builder.add_edge('fix_segment_boundaries_and_speakers', 'merge_same_speaker_neighbors')
        builder.add_edge('merge_same_speaker_neighbors', 'split_segments')
        builder.add_edge('split_segments', 'extract_relevant_segments')
        builder.add_edge('detect_speakers', 'fix_grammar')

        builder.add_edge('fix_grammar', 'construct_fixed_transcript')
        builder.add_edge('construct_fixed_transcript', END)
        return builder.compile()

    @staticmethod
    def _merge_timecodes(tc1: Timecodes | None, tc2: Timecodes | None) -> Timecodes | None:
        if tc1 is None and tc2 is None:
            return None
        start_candidates = [tc.start_time for tc in (tc1, tc2) if tc and tc.start_time is not None]
        end_candidates = [tc.end_time for tc in (tc1, tc2) if tc and tc.end_time is not None]
        return Timecodes(
            start_time=min(start_candidates) if start_candidates else None,
            end_time=max(end_candidates) if end_candidates else None,
        )

    @staticmethod
    def _merge_text(text_a: str, text_b: str) -> str:
        text_a = text_a.rstrip()
        text_b = text_b.lstrip()
        if not text_a:
            return text_b
        if not text_b:
            return text_a
        return f'{text_a} {text_b}'

    def merge_same_speaker_neighbors(self, state: FixedState) -> dict[str, list[Segment]]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Merging neighboring same-speaker segments ...')
        merged_segments: list[Segment] = []
        seg_idx = 0
        for seg in state.segments_with_fixed_boundaries_and_speakers:
            if not merged_segments or merged_segments[-1].speaker_id != seg.speaker_id:
                merged_seg = Segment.from_source(seg, _id=f'seg_{seg_idx}')
                merged_segments.append(merged_seg)
                seg_idx += 1
                continue

            prev = merged_segments[-1]
            prev.text = self._merge_text(prev.text, seg.text)
            prev.segment_references.add(seg.id)
            prev.timecodes = self._merge_timecodes(prev.timecodes, seg.timecodes)

        LOG.debug(f'[{state.source.id}][fix_transcript] Merged {len(state.segments_with_fixed_boundaries_and_speakers)}'
                  f' segments into {len(merged_segments)}.')
        return {'merged_segments': merged_segments}

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompt = read_text(f'{__package__}.resources.templates.fix_transcript', f'{name}.txt')
        return prompt

    def detect_speakers(self, state: FixedState) -> dict[str, list[Speaker]]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Detecting speakers ...')

        relevant_segments = '\n'.join(seg.model_dump_json() for seg in state.relevant_segments)
        prompt = self._load_prompt('detect_speakers').format(
            relevant_segments=relevant_segments,
            transcript_language=state.source.language,
        )
        detected_speakers = self.llm.get_completion(prompt, structured_output_type=SpeakerList).speakers

        LOG.debug(f'[{state.source.id}][fix_transcript] Detected speakers: {detected_speakers}')
        return {'detected_speakers': detected_speakers}

    @staticmethod
    def set_timecodes_from_source(state: FixedState, new_segments: list[Segment]) -> list[Segment]:
        source_segments_by_id = {s.id: s for s in state.source.segments}
        for segment in new_segments:
            segment.timecodes.start_time = min(
                [source_segments_by_id[ref].timecodes.start_time for ref in segment.segment_references])
            segment.timecodes.end_time = max(
                [source_segments_by_id[ref].timecodes.end_time for ref in segment.segment_references])
        return new_segments

    @staticmethod
    def _apply_boundary_corrections(segments: list[SegmentSource],
                                    corrections: list[SegmentBoundaryCorrection]) -> list[SegmentSource]:
        segment_by_id = {s.id: s for s in segments}
        correction_map = {c.id: c for c in corrections}

        for corr_id in correction_map:
            if corr_id not in segment_by_id:
                raise ValueError(f'Segment ID: {corr_id} in corrections not found in provided segments.')

        fixed_segments: list[SegmentSource] = []
        for segment in segments:
            if corr := correction_map.get(segment.id):
                update = {}
                if corr.corrected_text is not None:
                    update['text'] = corr.corrected_text
                if corr.speaker_id is not None:
                    update['speaker_id'] = corr.speaker_id
                fixed_segments.append(segment.model_copy(update=update))
            else:
                fixed_segments.append(segment.model_copy())
        return fixed_segments

    def fix_segment_boundaries_and_speakers(self, state: FixedState) -> dict[str, list[SegmentSource]]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Fixing segment boundaries and speakers ...')

        segments_str = '\n'.join(seg.model_dump_json(exclude={'segment_references'}) for seg in state.source.segments)
        prompt = self._load_prompt('fix_segment_boundaries_and_speakers').format(source=segments_str)
        corrections = self.llm.get_completion(
            prompt, structured_output_type=SegmentBoundaryCorrectionList).corrections
        out_segments = self._apply_boundary_corrections(state.source.segments, corrections)

        LOG.debug(f'[{state.source.id}][fix_transcript] Segments boundaries and speakers corrections: {corrections}')
        return {'segments_with_fixed_boundaries_and_speakers': out_segments}

    @staticmethod
    def _extract_segments_by_ids(segments: list[Segment], ids: set[str]) -> list[Segment]:
        segment_by_id: dict[str, Segment] = {s.id: s for s in segments}
        found_segments: list[Segment] = []
        for _id in ids:
            if (segment := segment_by_id.get(_id)) is not None:
                found_segments.append(segment)
            else:
                raise ValueError(f'Segment ID: {_id} references unknown segment!')
        return found_segments

    def extract_relevant_segments(
            self, state: FixedState) -> Command[Literal['extract_relevant_segments', 'split_segments']]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Extracting relevant segments ...')

        if state.relevant_segments_status is None:
            status = 'This is first call of this step, no errors encountered so far.'
        else:
            status = (f'This is attempt number {state.relevant_segments_status[0] + 1}, previous encountered error was'
                      f' {state.relevant_segments_status[1]}.')
        segments_str = '\n'.join(seg.model_dump_json(exclude={'segment_references'}) for seg in
                                 state.split_segments)
        prompt = self._load_prompt('select_relevant_segments').format(segments=segments_str, status=status)
        relevant_segment_ids = self.llm.get_completion(prompt, structured_output_type=RelevantSegmentIds)

        try:
            relevant_segments = self._extract_segments_by_ids(
                segments=state.split_segments,
                ids=relevant_segment_ids.relevant_segments)
            LOG.debug(f'[{state.source.id}][fix_transcript] Relevant segments IDs: {relevant_segment_ids}')
            # Sort the segments using timestamps
            relevant_segments = self._sort_segments(relevant_segments, state)
            return Command(update={'relevant_segments': relevant_segments}, goto='detect_speakers')
        except ValueError as e:
            LOG.warning(f'[{state.source.id}][fix_transcript] Relevant segments extraction failed with following error:'
                        f' {e}\nRetrying ...')
            status = state.relevant_segments_status or (0, '')
            status = status[0] + 1, str(e)
            if status[0] == 4:
                raise ValueError(f'[{state.source.id}][fix_transcript] Relevant segments extraction failed 3 times in'
                                 f' a row, last error: {e}\nExiting!')
            return Command(update={'relevant_segments_status': status}, goto='extract_relevant_segments')

    @staticmethod
    def _sort_segments(fixed_segments: list[Segment], state: FixedState) -> list[Segment]:
        source_order: dict[str, int] = {segment.id: idx for idx, segment in enumerate(state.source.segments)}

        def _sorting_key(segment: Segment) -> float:
            if not segment.segment_references:
                raise ValueError(f'[{state.source.id}] Fixed segment {segment.id} has no segment references.')
            try:
                ref_idx = min((source_order[ref_segment] for ref_segment in segment.segment_references))
                return ref_idx
            except KeyError:
                raise ValueError(f'[{state.source.id}] Fixed segment {segment.id} has invalid reference among'
                                 f' "{segment.segment_references}" references!')

        return sorted(fixed_segments, key=_sorting_key)

    @staticmethod
    def _get_timecodes_split(timecodes: Timecodes, segment_split: SegmentSplit) -> list[Timecodes | None]:
        if timecodes is None:
            return [None for _ in range(len(segment_split.text_parts))]
        timecodes_len = timecodes.end_time - timecodes.start_time
        text_parts_len = [len(text_part.text) for text_part in segment_split.text_parts]
        text_parts_est_time = [timecodes_len * text_part_len / sum(text_parts_len) for text_part_len in text_parts_len]

        split_timecodes: list[Timecodes] = []
        current_time = timecodes.start_time
        for est_time in text_parts_est_time:
            split_timecodes.append(Timecodes(start_time=current_time, end_time=current_time + est_time))
            current_time += est_time
        return split_timecodes

    def _apply_segment_splits(self, state: FixedState, segment_splits: list[SegmentSplit],
                              segments: list[Segment]) -> list[Segment]:
        splits_by_id = {s.id: s for s in segment_splits}
        out_segments: list[Segment] = []
        for segment in segments:
            if split := splits_by_id.pop(segment.id, None):
                timecodes_split = self._get_timecodes_split(segment.timecodes, split)
                for idx, text_part in enumerate(split.text_parts):
                    _id = f'{segment.id}_{idx}'
                    new_segment = Segment(id=_id, text=text_part.text, speaker_id=text_part.speaker_id,
                                          timecodes=timecodes_split[idx], segment_references=segment.segment_references)
                    out_segments.append(new_segment)
            else:
                out_segments.append(segment)
        if splits_by_id:
            raise ValueError(f'[{state.source.id}][fix_transcript] Segment splits contain unknown segment references:'
                             f' {splits_by_id}.')
        return out_segments

    def split_segments(self, state: FixedState) -> dict[str, list[Segment]]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Splitting some segments ...')

        merged_segments = '\n'.join(s.model_dump_json() for s in state.merged_segments)
        speakers_source = state.source.speakers or []
        speakers = '\n'.join(s.model_dump_json() for s in speakers_source)
        prompt = self._load_prompt('split_by_questions').format(relevant_segments=merged_segments, speakers=speakers)
        segment_splits = self.llm.get_completion(prompt, structured_output_type=SegmentSplitList).segment_splits

        ready_segments = self._apply_segment_splits(state, segment_splits, state.merged_segments)
        LOG.debug(f'[{state.source.id}][fix_transcript] Segment splits: {segment_splits}')
        return {'split_segments': ready_segments}

    @staticmethod
    def _apply_grammar_fixes(segments: list[Segment], corrected_texts: list[CorrectedText]) -> list[Segment]:
        corrected_map = {s.id: s.corrected_text for s in corrected_texts}
        return [s.model_copy(update={'text': corrected_map.get(s.id) or s.text}) for s in segments]

    def fix_grammar(self, state: FixedState) -> dict[str, list[Segment]]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Fixing text in segments ...')

        split_segments = '\n'.join([seg.model_dump_json() for seg in state.relevant_segments])
        prompt = self._load_prompt('fix_text_mistakes').format(relevant_segments=split_segments)
        corrected_texts = self.llm.get_completion(prompt, structured_output_type=CorrectedTextList).corrected_texts

        fixed_segments = self._apply_grammar_fixes(segments=state.relevant_segments, corrected_texts=corrected_texts)
        LOG.debug(f'[{state.source.id}][fix_transcript] Text corrections: {corrected_texts}')
        return {'fixed_segments': fixed_segments}

    @staticmethod
    def validate_fixed_transcript(original: Transcript, candidate: Transcript) -> None:
        """
        Compares original transcript and fixed one, generated by LLM, all checks are currently only soft, so they do not
        raise errors, but only logs warnings.

        :param original: Input `Transcript` generated by transcription system (Amazon Transcribe, for example)
        :param candidate: `Transcript` generated by LLM by fixing mistakes in the original one
        """
        if candidate.id != original.id:
            LOG.warning(f'[{original.id}][fix_transcript] Transcript id changed from '
                        f'"{original.id}" to "{candidate.id}"')
        if candidate.language != original.language:
            LOG.warning(f'[{original.id}][fix_transcript] Transcript language changed from '
                        f'"{original.language}" to "{candidate.language}"')

        if not candidate.segments:
            LOG.warning(f'[{original.id}][fix_transcript] Fixed transcript must contain at least one segment')

        speaker_ids = {speaker.speaker_id for speaker in candidate.speakers or []}
        if not speaker_ids:
            LOG.warning(f'[{original.id}][fix_transcript]  Fixed transcript must declare at least one speaker')

        seen_segments: set[str] = set()
        last_end: float | None = None
        for segment in candidate.segments:
            if segment.id in seen_segments:
                LOG.warning(f'[{original.id}][fix_transcript] Duplicate segment id "{segment.id}" in fixed transcript')
            seen_segments.add(segment.id)

            if segment.speaker_id not in speaker_ids:
                LOG.warning(f'[{original.id}][fix_transcript] Segment "{segment.id}" references unknown '
                            f'speaker "{segment.speaker_id}"')

            if segment.timecodes is None:
                continue
            start = segment.timecodes.start_time
            end = segment.timecodes.end_time
            if not start or not end:
                continue
            if start > end:
                LOG.warning(f'[{original.id}][fix_transcript] Segment "{segment.id}" has start_time {start} '
                            f'after end_time {end}')
            if last_end is not None and start < last_end:
                LOG.warning(f'[{original.id}][fix_transcript] Segment "{segment.id}" starts at {start} '
                            f'before previous end {last_end}')
            last_end = end

    def construct_fixed_transcript(self, state: FixedState) -> dict[str, Transcript]:
        LOG.debug(f'[{state.source.id}][fix_transcript] Creating final Transcript object ...')

        fixed_transcript = Transcript(id=state.source.id,
                                      language=state.source.language,
                                      url=state.source.url,
                                      speakers=state.detected_speakers,
                                      segments=state.fixed_segments)
        self.validate_fixed_transcript(state.source, fixed_transcript)
        LOG.debug(f'[{state.source.id}][fix_transcript] Fixed transcript: {fixed_transcript}')
        return {'fixed': fixed_transcript}
