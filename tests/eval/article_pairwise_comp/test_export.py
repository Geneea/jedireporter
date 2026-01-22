from pathlib import Path
from typing import Iterator
from unittest.mock import call, Mock

import pytest
from jedireporter.utils.gdoc import GoogleDocRequestsWrapper

from jedireporter.article import Article, Paragraph, Topic, TopicList
from jedireporter.eval.article_pairwise_comp.export import (
    ComparisonGoogleDocExporter,
    ComparisonMarkdownExporter,
)
from jedireporter.eval.article_pairwise_comp.models import (
    ArticleComparisonResult,
    ArticlePair,
    CriterionAssessment,
    CriterionOutcome,
    CriterionScore,
)
from jedireporter.impex import GoogleDocExporter
from jedireporter.transcript import SegmentSource, Speaker, Transcript


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        id='tr',
        language='en',
        segments=[
            SegmentSource(id='s2', text='Question', speaker_id='spk_0'),
            SegmentSource(id='s3', text='Answer', speaker_id='spk_1'),
        ],
        speakers=[
            Speaker(speaker_id='spk_0', role='host', name='Moderator', description=None),
            Speaker(speaker_id='spk_1', role='guest', name='Guest', description=None),
        ],
    )


@pytest.fixture
def article_model_a(transcript: Transcript) -> Article:
    return Article(
        id='article',
        language='en',
        paragraphs=[
            Paragraph(id='a-title-1', type='title', text='Article A Title', segment_refs=[]),
            Paragraph(id='a-text-1', type='text', text='Intro A', topic_id='topic_1', segment_refs=[]),
            Paragraph(id='a-question-1', type='question', text='Question A?', topic_id='topic_2',
                      speaker='spk_0', segment_refs=[]),
            Paragraph(id='a-answer-1', type='answer', text='Answer A', topic_id='topic_2', speaker='spk_1',
                      segment_refs=[]),
        ],
        transcript=transcript,
        topics=TopicList(topics=[Topic(id='topic_1', title='Topic 1'), Topic(id='topic_2', title='Topic 2')]),
        debug_info={'model': 'A'}
    )


@pytest.fixture
def article_model_b(transcript: Transcript) -> Article:
    return Article(
        id='article',
        language='en',
        paragraphs=[
            Paragraph(id='b-title-1', type='title', text='Article B Title', segment_refs=[]),
            Paragraph(id='b-title-2', type='title', text='Article B Alt', segment_refs=[]),
            Paragraph(id='b-text-1', type='text', text='Intro B', topic_id='topic_1', segment_refs=[]),
            Paragraph(id='b-question-1', type='question', text='Second question?', topic_id='topic_2',
                      speaker='spk_0', segment_refs=[]),
            Paragraph(id='b-answer-1', type='answer', text='B answer', topic_id='topic_2', speaker='spk_1',
                      segment_refs=[]),
        ],
        transcript=transcript,
        topics=TopicList(topics=[Topic(id='topic_1', title='Topic 1'), Topic(id='topic_2', title='Topic 2')]),
        debug_info={'model': 'B'}
    )


@pytest.fixture
def criterion_assessment() -> CriterionAssessment:
    return CriterionAssessment(
        id=1,
        name='Accuracy|Precision',
        article_a_score=CriterionScore.FULLY,
        article_b_score=CriterionScore.PARTLY,
        better=CriterionOutcome.ARTICLE_A,
        justification='Stronger reason\nLine 2 with | char',
    )


@pytest.fixture
def comparison_result(criterion_assessment: CriterionAssessment) -> ArticleComparisonResult:
    return ArticleComparisonResult(
        id='cmp-001',
        criteria=[criterion_assessment],
        winner='article_a',
        justification='Top line\nSecond line',
        confidence=0.876,
    )


@pytest.fixture
def article_pair(article_model_a: Article, article_model_b: Article, transcript: Transcript) -> ArticlePair:
    return ArticlePair(
        index=1,
        model_a_article=('ModelA', article_model_a),
        model_b_article=('ModelB', article_model_b),
        transcript=transcript,
    )


class TestComparisonMarkdownExporter:
    def test_article_to_markdown_lines(self, article_model_b: Article) -> None:
        lines = ComparisonMarkdownExporter._article_to_markdown_lines(article_model_b)
        assert lines == [
            '### Article B Title',
            '#### Alternative Titles',
            '- **Article B Alt**',
            '',
            '---',
            '##### Speakers',
            '',
            '- Moderator (host)',
            '- Guest (guest)',
            '---',
            '##### Summary',
            '',
            '---',
            '',
            '##### Topic 1',
            '',
            'Intro B',
            '',
            '',
            '##### Topic 2',
            '',
            '**Moderator:** Second question?',
            '',
            '**Guest:** "B answer"',
            '',
            '---',
            '##### Notes',
            ''
        ]

    def test_write_markdown(self, tmp_path: Path, article_model_a: Article, article_model_b: Article,
                            comparison_result: ArticleComparisonResult, article_pair: ArticlePair) -> None:
        exporter = ComparisonMarkdownExporter(tmp_path)

        exporter.write_markdown(article_pair, comparison_result)

        markdown_path = tmp_path / 'cmp-001.md'
        expected_text = (
            '# cmp-001\n'
            '\n'
            '**Winner:** Article A (ModelA)\\\n'
            '**Confidence:** 0.88\\\n'
            '**Justification:** Top line<br>Second line\n'
            '\n'
            '| Criterion | Article A (ModelA) | Article B (ModelB) | Better | Justification |\n'
            '| --- | --- | --- | --- | --- |\n'
            '| 1. Accuracy\\|Precision | fully | partly | article_a | Stronger reason<br>Line 2 with \\| char |\n'
            '\n'
            '### Article A (ModelA)\n'
            '### Article A Title\n'
            '\n'
            '---\n'
            '##### Speakers\n'
            '\n'
            '- Moderator (host)\n'
            '- Guest (guest)\n'
            '---\n'
            '##### Summary\n'
            '\n'
            '---\n'
            '\n'
            '##### Topic 1\n'
            '\n'
            'Intro A\n'
            '\n'
            '\n'
            '##### Topic 2\n'
            '\n'
            '**Moderator:** Question A?\n'
            '\n'
            '**Guest:** "Answer A"\n'
            '\n'
            '---\n'
            '##### Notes\n'
            '\n'
            '\n'
            '### Article B (ModelB)\n'
            '### Article B Title\n'
            '#### Alternative Titles\n'
            '- **Article B Alt**\n'
            '\n'
            '---\n'
            '##### Speakers\n'
            '\n'
            '- Moderator (host)\n'
            '- Guest (guest)\n'
            '---\n'
            '##### Summary\n'
            '\n'
            '---\n'
            '\n'
            '##### Topic 1\n'
            '\n'
            'Intro B\n'
            '\n'
            '\n'
            '##### Topic 2\n'
            '\n'
            '**Moderator:** Second question?\n'
            '\n'
            '**Guest:** "B answer"\n'
            '\n'
            '---\n'
            '##### Notes\n'
            '\n'
        )
        assert markdown_path.read_text() == expected_text


class TestComparisonGoogleDocExporter:
    @pytest.fixture
    def request_wrapper(self) -> Mock:
        return Mock(spec=GoogleDocRequestsWrapper)

    @pytest.fixture
    def comparison_google_doc_exporter(
            self, request_wrapper: GoogleDocRequestsWrapper) -> Iterator[ComparisonGoogleDocExporter]:
        exporter = ComparisonGoogleDocExporter(request_wrapper=request_wrapper)
        yield exporter

    @pytest.fixture
    def article(self, request) -> Article:
        return request.getfixturevalue(request.param)

    def test_add_summary_lines(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            comparison_result: ArticleComparisonResult,
    ) -> None:
        comparison_google_doc_exporter._add_summary_lines(comparison_result)

        expected_text = (
            'cmp-001\n'
            'Winner: article_a\n'
            'Confidence: 0.88\n'
            'Justification: Top line\n'
            'Second line\n'
        )
        request_wrapper.insert_text_request.assert_called_once_with(expected_text)
        request_wrapper.update_paragraph_style_request.assert_called_once_with(
            texts=[comparison_result.id], named_style='HEADING_1')
        request_wrapper.update_text_style_request.assert_called_once_with(
            texts=['Winner', 'Confidence', 'Justification'],
            text_style='bold',
            skip_formatted_paragraphs=True,
        )

    def test_add_criteria_table(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            comparison_result: ArticleComparisonResult,
    ) -> None:
        comparison_google_doc_exporter._add_criteria_table(comparison_result)
        request_wrapper.insert_table_request.assert_called_once_with(cols_nbr=5, rows_nbr=2)

    def test_fill_criteria_table_rows(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            article_pair: ArticlePair,
            comparison_result: ArticleComparisonResult,
    ) -> None:
        comparison_google_doc_exporter._fill_criteria_table_rows(article_pair, comparison_result)
        expected_calls = [
            call(0, 0, 0, 'Criterion'),
            call(0, 0, 1, 'Article A [ModelA]'),
            call(0, 0, 2, 'Article B [ModelB]'),
            call(0, 0, 3, 'Better'),
            call(0, 0, 4, 'Justification'),
            call(0, 1, 0, '1. Accuracy|Precision'),
            call(0, 1, 1, 'fully'),
            call(0, 1, 2, 'partly'),
            call(0, 1, 3, 'article_a'),
            call(0, 1, 4, 'Stronger reason\nLine 2 with | char'),
        ]
        request_wrapper.fill_table_cell_value_request.assert_has_calls(expected_calls)

    def test_make_criterion_table_header_bold(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            article_pair: ArticlePair,
    ) -> None:
        request_wrapper.get_ith_table_doc_range.return_value = (10, 25)
        comparison_google_doc_exporter._make_criterion_table_header_bold(article_pair)
        request_wrapper.update_text_style_request.assert_called_once_with(
            texts=['Criterion', 'Article A [ModelA]', 'Article B [ModelB]', 'Better', 'Justification'],
            text_style='bold',
            search_index_start=10,
            skip_formatted_paragraphs=True,
        )

    @pytest.mark.parametrize(
        'article, expected_output',
        [
            pytest.param('article_model_a', {'Moderator:', 'Guest:'}, id='model_a'),
            pytest.param('article_model_b', {'spk_0:', 'spk_1:'}, id='model_b'),
        ],
        indirect=['article'],
    )
    def test_get_speaker_set_for_article(
            self,
            article: Article,
            expected_output: set[str],
            comparison_google_doc_exporter: ComparisonGoogleDocExporter) -> None:
        if article.debug_info.get('model') == 'B':
            # Artificially remove the speakers for article b, to check the speakers directly from the Article object
            article.transcript.speakers = None
        speakers = comparison_google_doc_exporter._get_speaker_set_for_article(article)
        assert speakers == expected_output

    def test_make_all_qa_section_speakers_bold(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            article_pair: ArticlePair,
    ) -> None:
        request_wrapper.get_ith_table_doc_range.return_value = (42, 60)
        comparison_google_doc_exporter._make_all_qa_section_speakers_bold(article_pair)
        args, kwargs = request_wrapper.update_text_style_request.call_args

        assert kwargs['text_style'] == 'bold'
        assert kwargs['search_index_start'] == 42
        assert kwargs['find_all'] is True
        assert kwargs['skip_formatted_paragraphs'] is True
        assert set(kwargs['texts']) == {'Moderator:', 'Guest:'}

    def test_add_links(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
    ) -> None:
        doc_exporter = GoogleDocExporter()
        doc_exporter.sources = [('- Source A', 'https://a.example'), ('- Source B', 'https://b.example')]
        article = Article(
            id='article',
            language='en',
            paragraphs=[Paragraph(id='p1', type='title', text='Title', segment_refs=[])],
            topics=TopicList(topics=[Topic(id='topic', title='Topic')]),
        )

        comparison_google_doc_exporter._add_links(doc_exporter, article, search_index_start=5)

        expected_calls = [
            call(
                texts=['- Source A'],
                url_link='https://a.example',
                find_all=True,
                skip_formatted_paragraphs=True,
                search_index_start=5,
            ),
            call(
                texts=['- Source B'],
                url_link='https://b.example',
                find_all=True,
                skip_formatted_paragraphs=True,
                search_index_start=5,
            ),
        ]
        request_wrapper.update_text_style_request.assert_has_calls(expected_calls)

    def test_set_notes_section_style(
            self,
            comparison_google_doc_exporter: ComparisonGoogleDocExporter,
            request_wrapper: Mock,
            article_pair: ArticlePair,
    ) -> None:
        exporter_a = GoogleDocExporter()
        exporter_a.note_headings = ['AI Note 1', 'AI Note 2']
        exporter_b = GoogleDocExporter()
        exporter_b.note_headings = ['AI Note 1']
        comparison_google_doc_exporter.article_exporters = {'a': exporter_a, 'b': exporter_b}
        comparison_google_doc_exporter._add_links = Mock()
        request_wrapper.get_cell_index_from_table.side_effect = [10, 20]

        comparison_google_doc_exporter._set_notes_section_style(article_pair)

        expected_calls = [
            call(
                texts=['Notes'],
                named_style='HEADING_4',
                search_index_start=10,
                skip_formatted_paragraphs=True,
            ),
            call(
                texts=['Notes'],
                named_style='HEADING_4',
                search_index_start=20,
                skip_formatted_paragraphs=True,
            ),
            call(
                texts=['AI Note 1', 'AI Note 2'],
                named_style='HEADING_5',
                search_index_start=10,
            ),
            call(
                texts=['AI Note 1'],
                named_style='HEADING_5',
                search_index_start=20,
            ),
        ]
        request_wrapper.update_paragraph_style_request.assert_has_calls(expected_calls)
        request_wrapper.commit_requests.assert_called_once()
        comparison_google_doc_exporter._add_links.assert_has_calls([
            call(exporter_a, article_pair.model_a_article[1], search_index_start=10),
            call(exporter_b, article_pair.model_b_article[1], search_index_start=20),
        ])
