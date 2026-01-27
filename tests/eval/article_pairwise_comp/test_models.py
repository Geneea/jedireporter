from jedireporter.eval.article_pairwise_comp.models import (
    ArticleComparisonAggregator,
    ArticleComparisonResult,
    ArticleComparisonSummary,
    CriterionAssessment,
    CriterionOutcome,
    CriterionScore,
)


def test_aggregator_summary_counts_winners_and_ties() -> None:
    criterion_assessment = CriterionAssessment(
        id=1,
        name='Accuracy',
        article_a_score=CriterionScore.FULLY,
        article_b_score=CriterionScore.PARTLY,
        better=CriterionOutcome.ARTICLE_A,
        justification='j',
    )

    aggregator = ArticleComparisonAggregator()
    aggregator.add(ArticleComparisonResult(
        id='1', criteria=[criterion_assessment], winner='article_a', justification='j', confidence=0.5))
    aggregator.add(ArticleComparisonResult(
        id='2', criteria=[criterion_assessment], winner='article_b', justification='j', confidence=0.6))
    aggregator.add(ArticleComparisonResult(
        id='3', criteria=[criterion_assessment], winner='tie', justification='j', confidence=0.7))
    aggregator.add(ArticleComparisonResult(
        id='4', criteria=[criterion_assessment], winner='article_a', justification='j', confidence=0.4))

    summary = aggregator.summary

    assert summary is not None
    assert summary == ArticleComparisonSummary(id='summary', total=4, ties=1,
                                               wins_per_model={
                                                   'article_a': 2,
                                                   'article_b': 1,
                                                   'tie': 1
                                               })
