from collections import defaultdict
from enum import Enum
from typing import Literal

from pydantic import BaseModel, computed_field, Field, model_validator

from jedireporter.article import Article
from jedireporter.camelModel import CamelModel
from jedireporter.transcript import Transcript


class ArticlePair(BaseModel):
    index: int
    model_a_article: tuple[str, Article]  # model name + generated article
    model_b_article: tuple[str, Article]  # model name + generated article
    transcript: Transcript | None

    @property
    def pair_id(self) -> str:
        return self.model_a_article[1].id

    @model_validator(mode='after')
    def verify_article_ids(self) -> 'ArticlePair':
        if self.model_a_article[1].id != self.model_b_article[1].id:
            raise ValueError(f'Article id mismatch (A: {self.model_a_article[1].id}, B: {self.model_b_article[1].id})'
                             f' at index {self.index}.')
        else:
            return self


class CriterionScore(str, Enum):
    FULLY = 'fully'
    PARTLY = 'partly'
    NOT_AT_ALL = 'not_at_all'
    NOT_EVALUATED = 'not_evaluated'


class CriterionOutcome(str, Enum):
    ARTICLE_A = 'article_a'
    ARTICLE_B = 'article_b'
    TIE = 'tie'
    NOT_EVALUATED = 'not_evaluated'


class CriterionAssessment(CamelModel):
    id: int = Field(description='Numeric identifier of the evaluated guideline (1-based).')
    name: str = Field(description='Short name of the guideline/criterion.')
    article_a_score: CriterionScore = Field(description='How Article A satisfies the guideline.')
    article_b_score: CriterionScore = Field(description='How Article B satisfies the guideline.')
    better: CriterionOutcome = Field(
        description='Which article performed better for this guideline, or not_evaluated when not scored.')
    justification: str = Field(description='Brief explanation referencing concrete evidence for the scores.')


class ArticleComparisonVerdict(CamelModel):
    criteria: list[CriterionAssessment] = Field(
        description='Detailed assessments for every evaluation criterion.')
    winner: Literal['article_a', 'article_b', 'tie'] = Field(
        description='Preferred article based on the aggregated criterion results.')
    justification: str = Field(
        description='Summary explaining how the criterion outcomes led to the final decision.')
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description='Confidence in the verdict on a 0.0–1.0 scale (lower values mean greater uncertainty).',
    )


class ArticleComparisonResult(CamelModel):
    id: str
    criteria: list[CriterionAssessment]
    winner: str
    justification: str
    confidence: float | None


class ArticleComparisonSummary(CamelModel):
    id: str
    total: int
    wins_per_model: dict[str, int]
    ties: int


class ArticleComparisonAggregator(CamelModel):
    per_sample: list[ArticleComparisonResult] = Field(default_factory=list)

    def add(self, result: ArticleComparisonResult) -> None:
        self.per_sample.append(result)

    @computed_field
    @property
    def summary(self) -> ArticleComparisonSummary | None:
        if not self.per_sample:
            return None

        total = len(self.per_sample)
        wins_per_model = defaultdict(int)
        for result in self.per_sample:
            wins_per_model[result.winner] += 1
        ties = sum(1 for result in self.per_sample if result.winner == 'tie')

        return ArticleComparisonSummary(
            id='summary',
            total=total,
            wins_per_model=wins_per_model,
            ties=ties
        )
