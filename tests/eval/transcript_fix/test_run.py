from pathlib import Path

import pydantic_core
import pytest

from jedireporter.eval.transcript_fix.run_transcript_eval import EvaluationConfig, TranscriptFixEvaluation
from jedireporter.transcript import Transcript


def minimal_transcript(transcript_id: str) -> Transcript:
    return Transcript(id=transcript_id, language='en', segments=[])


def build_config(tmp_path: Path,
                 *,
                 source_transcripts: list[Transcript],
                 generated_transcripts: list[Transcript],
                 gold_transcripts: list[Transcript],
                 evaluate_missing_gold: bool) -> EvaluationConfig:
    def _generate_transcripts_str(transcripts: list[Transcript]) -> str:
        return '\n'.join(transcript.model_dump_json() for transcript in transcripts)

    source_path = tmp_path / 'source.jsonl'
    source_path.write_text(_generate_transcripts_str(source_transcripts))
    generated_path = tmp_path / 'generated.jsonl'
    generated_path.write_text(_generate_transcripts_str(generated_transcripts))
    gold_path = tmp_path / 'gold.jsonl'
    gold_path.write_text(_generate_transcripts_str(gold_transcripts))
    output_dir = tmp_path / 'outputs'
    return EvaluationConfig(
        gold_path=gold_path,
        source=source_path,
        generated=generated_path,
        output_dir=output_dir,
        evaluate_missing_gold=evaluate_missing_gold,
    )


class TestTranscriptFixEvaluation:
    def test_prepare_transcript_pairs(self, tmp_path: Path) -> None:
        transcript = minimal_transcript('Test_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[transcript],
            generated_transcripts=[transcript],
            gold_transcripts=[transcript],
            evaluate_missing_gold=True,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)
        transcript_pairs = list(transcript_fix_evaluator.prepare_transcript_pairs())

        assert len(transcript_pairs) == 1
        assert isinstance(transcript_pairs[0][2], Transcript)
        assert transcript_pairs[0][0].id == transcript_pairs[0][1].id == transcript_pairs[0][2].id

    def test_missing_source_transcript_raises(self, tmp_path: Path) -> None:
        transcript = minimal_transcript('Test_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[],
            generated_transcripts=[transcript],
            gold_transcripts=[transcript],
            evaluate_missing_gold=True,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)

        with pytest.raises(ValueError) as exc:
            list(transcript_fix_evaluator.prepare_transcript_pairs())
        assert 'Source file is missing transcript' in str(exc.value)
        assert 'Test_transcript.mp3' in str(exc.value)

    def test_mismatched_source_transcript_ids_raise(self, tmp_path: Path) -> None:
        source_transcript = minimal_transcript('Source_transcript.mp3')
        generated_transcript = minimal_transcript('Test_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[source_transcript],
            generated_transcripts=[generated_transcript],
            gold_transcripts=[generated_transcript],
            evaluate_missing_gold=True,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)

        with pytest.raises(ValueError) as exc:
            list(transcript_fix_evaluator.prepare_transcript_pairs())

        message = str(exc.value)
        assert 'Source transcript ID Source_transcript.mp3' in message
        assert 'Test_transcript.mp3' in message

    def test_missing_gold_transcript_allowed(self, tmp_path: Path) -> None:
        transcript = minimal_transcript('Test_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[transcript],
            generated_transcripts=[transcript],
            gold_transcripts=[],
            evaluate_missing_gold=True,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)
        transcript_pairs = list(transcript_fix_evaluator.prepare_transcript_pairs())

        assert len(transcript_pairs) == 1
        source_transcript, generated_transcript, gold_transcript = transcript_pairs[0]
        assert source_transcript.id == generated_transcript.id == 'Test_transcript.mp3'
        assert gold_transcript is None

    def test_missing_gold_transcript_disallowed(self, tmp_path: Path) -> None:
        transcript = minimal_transcript('Test_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[transcript],
            generated_transcripts=[transcript],
            gold_transcripts=[],
            evaluate_missing_gold=False,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)

        with pytest.raises(pydantic_core.ValidationError):
            list(transcript_fix_evaluator.prepare_transcript_pairs())

    def test_mismatched_gold_transcript_ids_evaluated_without_gold(self, tmp_path: Path) -> None:
        source_transcript = minimal_transcript('Test_transcript.mp3')
        generated_transcript = minimal_transcript('Test_transcript.mp3')
        gold_transcript = minimal_transcript('Gold_transcript.mp3')
        config = build_config(
            tmp_path,
            source_transcripts=[source_transcript],
            generated_transcripts=[generated_transcript],
            gold_transcripts=[gold_transcript],
            evaluate_missing_gold=False,
        )
        transcript_fix_evaluator = TranscriptFixEvaluation(config=config)
        transcript_pairs = list(transcript_fix_evaluator.prepare_transcript_pairs())

        assert len(transcript_pairs) == 1
        source_transcript, generated_transcript, gold_transcript = transcript_pairs[0]
        assert source_transcript.id == generated_transcript.id == 'Test_transcript.mp3'
        assert gold_transcript is None
