import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.resources import read_text
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable, Iterator, Literal, Type, TypeVar

from dotenv import load_dotenv

from jedireporter.article import Article
from jedireporter.eval.article_pairwise_comp.export import (
    ComparisonGoogleDocExporter,
    ComparisonMarkdownExporter
)
from jedireporter.eval.article_pairwise_comp.models import (
    ArticleComparisonAggregator,
    ArticleComparisonResult,
    ArticleComparisonSummary,
    ArticleComparisonVerdict,
    ArticlePair,
)
from jedireporter.llm import InstructorLLM, LLMProfileLoader
from jedireporter.transcript import Transcript
from jedireporter.utils import logging as logutil
from jedireporter.utils import parallel

LOG = logutil.getLogger(__package__, __file__)

ModelT = TypeVar("ModelT", Article, Transcript)


@dataclass(frozen=True)
class EvaluationConfig:
    model_a_articles_path: Path
    model_b_articles_path: Path
    output_dir: Path
    llm_profile: str
    max_concurrency: int
    seed: int
    transcripts_path: Path | None
    google_service_credentials: Path | None
    google_token: Path | None
    google_folder: str | None


class ArticlePairwiseComparison:
    SYSTEM_PROMPT = read_text(
        'jedireporter.eval.article_pairwise_comp.resources',
        'article_pairwise_judge_system.txt',
        encoding='utf-8',
    )

    USER_PROMPT_TEMPLATE = read_text(
        'jedireporter.eval.article_pairwise_comp.resources',
        'article_pairwise_judge_user.txt',
        encoding='utf-8',
    )

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._random = random.Random(self.config.seed)
        self.aggregator = ArticleComparisonAggregator()
        self.markdown_exporter = ComparisonMarkdownExporter(self.config.output_dir)
        if self.config.google_token:
            self.google_doc_exporter = ComparisonGoogleDocExporter(token_path=self.config.google_token,
                                                                   google_folder=self.config.google_folder)
        elif self.config.google_service_credentials:
            self.google_doc_exporter = ComparisonGoogleDocExporter(
                service_credentials_path=self.config.google_service_credentials,
                google_folder=self.config.google_folder)
        else:
            raise ValueError('Must provide either --google-api-token or --google-api-credentials to access Google Docs '
                             'API!')
        self.model_a_name: str = self._resolve_model_name(self.config.model_a_articles_path)
        self.model_b_name: str = self._resolve_model_name(self.config.model_b_articles_path)

        profile = LLMProfileLoader.get(self.config.llm_profile)
        self._llm_profile = self.config.llm_profile
        self.llm = InstructorLLM.from_profile(profile)

    @staticmethod
    def _resolve_model_name(model_path) -> str:
        """Tries to identify one of known models in the filename, if not found returns full filename."""
        stem = model_path.stem
        for model in ['default', 'gpt-5-mini', 'gpt-5', 'mistral', 'gpt-4.1', 'claude']:
            if stem.find(model) != -1:
                if model == 'default':
                    return f'{model} (gpt-4.1)'
                return model
        return stem

    def prepare_article_pairs(self) -> Iterable[ArticlePair]:
        with self.config.model_a_articles_path.open(encoding='utf-8') as file_a, \
                self.config.model_b_articles_path.open(encoding='utf-8') as file_b:
            if self.config.transcripts_path:
                with self.config.transcripts_path.open(encoding='utf-8') as transcript_file:
                    yield from self._iterate_pairs(file_a, file_b, transcript_file)
            else:
                yield from self._iterate_pairs(file_a, file_b, None)

    @staticmethod
    def _load_to_mapping_by_id(file: TextIOWrapper, model: Type[ModelT]) -> dict[str, Article]:
        mapping: dict[str, Article] = {}
        for line in file:
            obj = model.model_validate_json(line)
            mapping[obj.id] = obj
        return mapping

    def _iterate_pairs(self, file_a, file_b, transcript_file) -> Iterator[ArticlePair]:
        transcript_mapping = self._load_to_mapping_by_id(transcript_file, Transcript) if transcript_file else {}
        article_model_a_mapping = self._load_to_mapping_by_id(file_a, Article)
        unused_model_b_articles = 0

        for index, line_b in enumerate(file_b):
            article_b = Article.model_validate_json(line_b)
            if article_a := article_model_a_mapping.pop(article_b.id, None):
                expected_id = article_b.id.removeprefix('article-')
                transcript = transcript_mapping.pop(expected_id, None)
                if transcript_mapping and transcript is None:
                    LOG.debug(f'Transcript with ID {expected_id} was not found, continuing without transcript.')
                yield ArticlePair(
                    index=index,
                    model_a_article=(self.model_a_name, article_a),
                    model_b_article=(self.model_b_name, article_b),
                    transcript=transcript,
                )
            else:
                LOG.debug(f'Article ID {article_b.id} from model {self.model_b_name} is not present in article '
                          f'IDs from model {self.model_a_name}. Skipping this article.')
                unused_model_b_articles += 1
                continue
        # Log to debug unused articles from model A
        for article_id in article_model_a_mapping:
            LOG.debug(f'Article ID {article_id} from model {self.model_a_name} was not present in article '
                      f'IDs from model {self.model_b_name}.')
        # Log to warning channel only sum of unused ids
        if unused_model_b_articles:
            LOG.warning(f'{unused_model_b_articles} loaded articles from model B were unused.')
        if article_model_a_mapping:
            LOG.warning(f'{len(article_model_a_mapping)} loaded articles from model A were unused.')
        if transcript_mapping:
            LOG.warning(f'{len(transcript_mapping)} loaded transcripts were unused.')

    @staticmethod
    def _get_article_headlines_count(article: Article) -> int:
        return sum([1 for p in article.paragraphs if p.type == 'title'])

    @staticmethod
    def _get_article_words_count(article: Article) -> int:
        return sum(len(paragraph.text.split()) for paragraph in article.paragraphs)

    def _build_prompt(self, pair: ArticlePair) -> str:
        transcript_section = 'Transcript not provided.'
        if pair.transcript is not None:
            transcript_section = 'Source Transcript:\n' + pair.transcript.model_dump_json(indent=2)
        article_a = pair.model_a_article[1]
        article_b = pair.model_b_article[1]
        return self.USER_PROMPT_TEMPLATE.format(
            article_a=article_a.model_dump_json(indent=2),
            article_a_words_count=self._get_article_words_count(article_a),
            article_a_headlines_count=self._get_article_headlines_count(article_a),
            article_b=article_b.model_dump_json(indent=2),
            article_b_words_count=self._get_article_words_count(article_b),
            article_b_headlines_count=self._get_article_headlines_count(article_b),
            transcript_section=transcript_section,
        )

    def _judge_pair(self, pair: ArticlePair) -> ArticleComparisonVerdict:
        prompt = self._build_prompt(pair)
        verdict = self.llm.get_completion(
            prompt,
            structured_output_type=ArticleComparisonVerdict,
            system_prompt=self.SYSTEM_PROMPT,
            strict=False
        )
        return verdict

    def _shuffle_pair(self, pair: ArticlePair) -> ArticlePair:
        if self._random.random() < 0.5:
            temp_model_article = pair.model_a_article
            pair.model_a_article = pair.model_b_article
            pair.model_b_article = temp_model_article
        return pair

    @staticmethod
    def _get_winner_model_name(pair: ArticlePair, llm_decision: Literal['article_a', 'article_b', 'tie']) -> str:
        if llm_decision == 'article_a':
            return pair.model_a_article[0]
        elif llm_decision == 'article_b':
            return pair.model_b_article[0]
        else:
            return 'tie'

    def _process_pair(self, pair: ArticlePair) -> tuple[ArticlePair, ArticleComparisonResult]:
        pair = self._shuffle_pair(pair)
        verdict = self._judge_pair(pair)
        result = ArticleComparisonResult(
            id=pair.pair_id,
            criteria=verdict.criteria,
            winner=self._get_winner_model_name(pair, verdict.winner),
            justification=verdict.justification,
            confidence=verdict.confidence,
        )
        return pair, result

    def run(self) -> None:
        pairs = self.prepare_article_pairs()

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            results = parallel.parallelMap(
                executor,
                self._process_pair,
                pairs,
                threadCount=self.config.max_concurrency,
            )

            LOG.info('Started pairwise article comparison')
            for pair_result in results:
                pair, result = pair_result
                self.aggregator.add(result)
                LOG.debug(f'[{result.id}] Winner: {result.winner}')
                self._write_output(result)
                self.google_doc_exporter.write_to_google_doc(pair, result)
                self.markdown_exporter.write_markdown(pair, result)

            summary = self.aggregator.summary
            if summary:
                self._write_output(summary)
                LOG.info(f'{self.model_a_name} wins: {summary.wins_per_model.get(self.model_a_name)}\n'
                         f'{self.model_b_name} wins: {summary.wins_per_model.get(self.model_b_name)}\n'
                         f'Ties: {summary.ties}')
            LOG.info('Finished pairwise article comparison')

    def _write_output(self, result: ArticleComparisonResult | ArticleComparisonSummary) -> None:
        output_dir = self.config.output_dir

        output_path = output_dir / f'{result.id}.json'
        with output_path.open('w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
            f.write('\n')


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description='Run LLM judge for pairwise article comparison experiments.')
    parser.add_argument('--model-a-articles',
                        help='Path to JSONL with Articles from model A',
                        required=True,
                        type=Path)
    parser.add_argument('--model-b-articles',
                        help='Path to JSONL with Articles from model B',
                        required=True,
                        type=Path)
    parser.add_argument('--transcripts',
                        help='Optional JSONL with source transcripts aligned with the articles',
                        required=False,
                        type=Path)
    parser.add_argument('--output-dir',
                        help='Directory to store evaluation results',
                        required=True,
                        type=Path)
    parser.add_argument('--llm-profile',
                        default='claude',
                        choices=['default', 'gpt-5', 'gpt-5-mini', 'claude', 'gpt-4.1'],
                        help='LLM profile name used for article comparison',
                        type=str)
    parser.add_argument('--max-concurrency',
                        type=int,
                        default=4,
                        help='Maximum number of parallel LLM invocations when a batch is provided')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for random number generator. Used for deciding the LLM model for article A/B.')
    parser.add_argument('--google-service-credentials',
                        type=Path,
                        help='Path to a JSON file with Google API credentials for service account.')
    parser.add_argument('--google-api-token',
                        type=Path,
                        help='Path to a file with access token to Google API')
    parser.add_argument('--google-folder',
                        type=str,
                        default=None,
                        help='ID of a folder on Google Drive for storing the generated articles to.')

    logutil.addLogArguments(parser)
    args = parser.parse_args()
    logutil.configureFromArgs(args)

    if not args.output_dir.exists():
        raise FileNotFoundError(f'Directory {args.output_dir} does not exist')

    return EvaluationConfig(
        model_a_articles_path=args.model_a_articles,
        model_b_articles_path=args.model_b_articles,
        transcripts_path=args.transcripts,
        output_dir=args.output_dir,
        llm_profile=args.llm_profile,
        max_concurrency=args.max_concurrency,
        google_service_credentials=args.google_service_credentials,
        google_token=args.google_api_token,
        seed=args.seed,
        google_folder=args.google_folder,
    )


def main() -> None:
    config = parse_args()

    load_dotenv()
    ArticlePairwiseComparison(config).run()


if __name__ == '__main__':
    main()
