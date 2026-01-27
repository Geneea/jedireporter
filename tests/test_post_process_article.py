from unittest.mock import Mock

import pytest

from jedireporter.article import (
    Article,
    ArticleMetadata,
    Paragraph,
    SegmentRef,
    Topic,
    TopicList,
    WebSearchResult,
    WebSearchResultList, WebSource,
)
from jedireporter.post_process_article import PostProcessArticleNodes, PostProcessArticleState
from jedireporter.transcript import SegmentSource, Timecodes, Transcript


@pytest.fixture
def intro_para() -> Paragraph:
    return Paragraph(
        id='p1',
        type='text',
        text='Intro',
        segment_refs=[SegmentRef(segment_id='s1'), SegmentRef(segment_id='s2')],
        topic_id='topic',
    )


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        id='tr',
        language='en',
        segments=[
            SegmentSource(id='s1', text='one', speaker_id='spk', timecodes=Timecodes(start_time=1, end_time=2)),
            SegmentSource(id='s2', text='two', speaker_id='spk', timecodes=Timecodes(start_time=5, end_time=7)),
        ],
        speakers=None,
    )


def test_get_referenced_timecodes(intro_para: Paragraph, transcript: Transcript) -> None:
    timecodes = PostProcessArticleNodes.get_referenced_timecodes(intro_para, transcript)
    assert timecodes == [Timecodes(start_time=1, end_time=2),
                         Timecodes(start_time=5, end_time=7)]


def test_add_source_timestamps(intro_para: Paragraph, transcript: Transcript) -> None:
    paragraph_without_refs = Paragraph(
        id='p2',
        type='text',
        text='Outro',
        segment_refs=[],
        topic_id='topic',
    )
    article = Article(
        id='article',
        language='en',
        paragraphs=[intro_para, paragraph_without_refs],
        topics=TopicList(topics=[Topic(id='topic', title='Topic')]),
    )
    state = PostProcessArticleState(article=article, transcript=transcript)
    nodes = PostProcessArticleNodes(llm=Mock())

    update = nodes.add_source_timestamps(state)

    updated = update['paragraphs_with_source_timestamps']
    assert updated[0].source_timecodes == Timecodes(start_time=1, end_time=7)
    assert updated[1].source_timecodes is None


def test_enrich_article(intro_para: Paragraph) -> None:
    article = Article(
        id='article',
        language='en',
        paragraphs=[intro_para],
        topics=TopicList(topics=[Topic(id='topic', title='Topic')]),
    )
    web_searches = WebSearchResultList(
        data=[
            WebSearchResult(
                id='acme',
                text_occurrences=[],
                query='Acme Corp',
                summary='Note',
                sources=[WebSource(url='https://example.com', title='Example', snippet='Example snippet')],
            )
        ]
    )
    enriched_paragraphs = [
        Paragraph(
            id='p1',
            type='text',
            text='Intro',
            segment_refs=[],
            topic_id='topic',
            source_timecodes=Timecodes(start_time=0, end_time=1),
        )
    ]
    state = PostProcessArticleState(
        article=article,
        transcript=Mock(Transcript),
        web_search_data=web_searches,
        paragraphs_with_source_timestamps=enriched_paragraphs,
    )

    update = PostProcessArticleNodes.enrich_article(state)

    enriched = update['enriched_article']
    assert enriched.metadata == ArticleMetadata(web_searches=web_searches)
    assert enriched.paragraphs == enriched_paragraphs
