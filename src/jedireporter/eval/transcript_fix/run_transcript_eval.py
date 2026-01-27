import argparse
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable

import pydantic_core

from jedireporter.eval.transcript_fix.metrics import MetricsCollector, TranscriptMetrics
from jedireporter.eval.transcript_fix.models import MetricsAggregator
from jedireporter.eval.transcript_fix.utils import TranscriptPair
from jedireporter.transcript import Transcript
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)


@dataclass(frozen=True)
class EvaluationConfig:
    gold_path: Path
    source: Path
    generated: Path
    output_dir: Path
    evaluate_missing_gold: bool


class TranscriptFixEvaluation:

    def __init__(self, config: EvaluationConfig) -> None:
        self.config: EvaluationConfig = config
        self.aggregator: MetricsAggregator = MetricsAggregator()
        self.per_sample_records: list[dict[str, float]] = []

    def prepare_transcript_pairs(self) -> Iterable[tuple[Transcript, Transcript, Transcript | None]]:
        with open(self.config.gold_path) as gold_file, open(self.config.source) as source_file, open(
                self.config.generated) as generated_file:
            # Pair lines from gold, source and generated using zip_longest:
            # - Stop when generated is exhausted (no more evaluation input)
            # - If source runs out first error is raised
            # - If gold runs out first or a gold line is invalid error is raised, optionally continue
            #   with 'no gold' when --evaluate-missing-gold is set
            for line_gold, line_source, line_generated in zip_longest(gold_file, source_file, generated_file,
                                                                      fillvalue=None):
                # Stop when generated is exhausted
                if line_generated is None:
                    break
                generated_transcript = Transcript.model_validate_json(line_generated)

                # Raise error when source is missing or IDs are mismatched
                if line_source is None:
                    raise ValueError(f'Source file is missing transcript for generated transcript '
                                     f'with ID: {generated_transcript.id}.')
                source_transcript = Transcript.model_validate_json(line_source)
                if source_transcript.id != generated_transcript.id:
                    raise ValueError(f'Source transcript ID {source_transcript.id} mismatches with generated transcript'
                                     f'ID {generated_transcript.id}.')

                try:
                    gold_transcript = Transcript.model_validate_json(line_gold)
                except pydantic_core.ValidationError as exc:
                    if self.config.evaluate_missing_gold:
                        gold_transcript = None
                        LOG.info(
                            f'Source transcript id {source_transcript.id} does not have golden reference, '
                            f'evaluating without gold sample.')
                    else:
                        raise exc

                if gold_transcript is not None and gold_transcript.id != generated_transcript.id:
                    # Mismatched ids: evaluate as candidate-only (no gold contribution to averages).
                    LOG.warning(
                        f'Gold transcript id {gold_transcript.id} does not match source transcript id '
                        f'{generated_transcript.id}, evaluating without gold sample!')
                    gold_transcript = None

                yield source_transcript, generated_transcript, gold_transcript

    @staticmethod
    def process_transcript(transcripts: tuple[Transcript, Transcript, Transcript | None]) -> MetricsCollector:
        source_transcript, fixed_transcript, gold_transcript = transcripts
        metrics = TranscriptMetrics(TranscriptPair(
            source=source_transcript,
            candidate=fixed_transcript,
            gold=gold_transcript,
        ))
        return metrics.compute()

    def run(self) -> None:
        transcripts = self.prepare_transcript_pairs()
        results = (self.process_transcript(transcript) for transcript in transcripts)

        LOG.info('Started')
        for result in results:
            self.aggregator.add(result)
            LOG.debug(f'[{result.id}] WER (Transcript-level; candidate x gold): '
                      f'{result.text_metrics.transcript_level.candidate2gold_wer}')
            self._write_output(result)
        summary = self.aggregator.summary
        self._write_output(summary)
        LOG.info(f'Saved evaluation artefacts to {self.config.output_dir.absolute()} folder')
        LOG.info(f'Average WER (Transcript-level; candidate x gold): '
                 f'{summary.text_metrics.transcript_level.candidate2gold_wer}')
        LOG.info('Finished')

    def _write_output(self, result: MetricsCollector) -> None:
        if not self.config.output_dir:
            return
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / f'{result.id}.json'
        with metrics_path.open('w', encoding='utf-8') as f:
            print(result.model_dump_json(indent=2), file=f)


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description='Run transcript quality evaluation experiments')
    parser.add_argument('--generated',
                        help='Path to JSONL with generated transcripts (skip workflow step)',
                        required=True,
                        type=Path)
    parser.add_argument('--gold',
                        default='gold_transcripts',
                        help='Path to JSONL with gold transcripts',
                        required=True,
                        type=Path)
    parser.add_argument('--source',
                        help='JSONL with original ASR transcripts used both as workflow input and baseline',
                        required=True,
                        type=Path)
    parser.add_argument('--output-dir',
                        help='Directory to store evaluation results',
                        required=True,
                        type=Path)
    parser.add_argument('--evaluate-missing-gold',
                        action='store_true',
                        help='Run evaluation without matching gold reference instead of failing.')
    logutil.addLogArguments(parser)
    args = parser.parse_args()
    logutil.configureFromArgs(args)

    if not args.output_dir.exists():
        raise FileNotFoundError(f'Directory {args.output_dir} does not exist.')

    return EvaluationConfig(
        generated=args.generated,
        gold_path=args.gold,
        source=args.source,
        output_dir=args.output_dir,
        evaluate_missing_gold=args.evaluate_missing_gold
    )


def main() -> None:
    config = parse_args()
    TranscriptFixEvaluation(config).run()


if __name__ == '__main__':
    main()
