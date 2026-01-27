from unittest.mock import call, Mock

import pytest

from jedireporter.article import (
    Article,
    ArticleMetadata,
    Paragraph,
    TextOccurrence,
    Topic,
    TopicList,
    WebSearchResult,
    WebSearchResultList,
    WebSource,
)
from jedireporter.impex import ExportHelper, GoogleDocExporter, MarkdownExporter
from jedireporter.transcript import SegmentSource, Speaker, Timecodes, Transcript


@pytest.fixture
def title_paragraph_1() -> Paragraph:
    return Paragraph(id='p-title-1', type='title', text='Main Title', segment_refs=[])


@pytest.fixture
def title_paragraph_2() -> Paragraph:
    return Paragraph(id='p-title-2', type='title', text='Alternative Title', segment_refs=[])


@pytest.fixture
def text_paragraph() -> Paragraph:
    return Paragraph(id='p-text-1', type='text', text='Intro', topic_id='topic', segment_refs=[])


@pytest.fixture
def question_paragraph() -> Paragraph:
    return Paragraph(id='p-question-1', type='question', text='What about Acme Corp?', topic_id='topic',
                     speaker='spk_1', segment_refs=[])


@pytest.fixture
def answer_paragraph() -> Paragraph:
    return Paragraph(id='p-answer-1', type='answer', text='Answer about Acme Corp.', topic_id='topic',
                     speaker='spk_2', segment_refs=[])


@pytest.fixture
def speaker_1() -> Speaker:
    return Speaker(speaker_id='spk_1', role='host', name='John Doe', description='Host')


@pytest.fixture
def speaker_2() -> Speaker:
    return Speaker(speaker_id='spk_2', role='guest', name='Max Mustermann', description='Guest')


@pytest.fixture
def topics() -> TopicList:
    return TopicList(topics=[Topic(id='topic', title='Topic')])


class TestExportHelper:
    def test_get_titles(self, title_paragraph_1: Paragraph, title_paragraph_2: Paragraph,
                        text_paragraph: Paragraph) -> None:
        article = Article(id='article-1', language='en',
                          paragraphs=[title_paragraph_1, title_paragraph_2, text_paragraph],
                          topics=TopicList(topics=[Topic(id='topic', title='Topic')]))
        helper = ExportHelper(article)

        main_title, alternative_titles = helper.get_titles()

        assert main_title == 'Main Title'
        assert alternative_titles == ['Alternative Title']

    def test_get_titles_main_missing(self, text_paragraph, question_paragraph) -> None:
        article = Article(id='article-2', language='en', paragraphs=[text_paragraph, question_paragraph],
                          topics=TopicList(topics=[Topic(id='topic', title='Topic')]))
        helper = ExportHelper(article)

        main_title, alternative_titles = helper.get_titles()

        assert main_title == 'article-2'
        assert alternative_titles == []

    @pytest.mark.parametrize(
        ('include_transcript', 'expected_speakers'),
        [
            (True, ['John Doe (Host)', 'Max Mustermann (Guest)']),
            (False, ['spk_1', 'spk_2']),
        ],
    )
    def test_get_speakers(
            self,
            title_paragraph_1: Paragraph,
            question_paragraph: Paragraph,
            answer_paragraph: Paragraph,
            text_paragraph: Paragraph,
            speaker_1: Speaker,
            speaker_2: Speaker,
            include_transcript: bool,
            expected_speakers: list[str],
    ) -> None:
        transcript = None
        if include_transcript:
            segments = [
                SegmentSource(id='s0', text='segment-0', speaker_id=speaker_1.speaker_id),
                SegmentSource(id='s1', text='segment-1', speaker_id=speaker_2.speaker_id),
            ]
            transcript = Transcript(id='transcript-1', language='en', segments=segments,
                                    speakers=[speaker_1, speaker_2])
        article = Article(
            id='article-3',
            language='en',
            paragraphs=[title_paragraph_1, question_paragraph, answer_paragraph, text_paragraph],
            transcript=transcript,
            topics=TopicList(topics=[Topic(id='topic', title='Topic')])
        )
        helper = ExportHelper(article)

        speakers = list(helper.get_speakers())

        assert set(speakers) == set(expected_speakers)


class TestMarkdownExporter:
    def test_get_titles(self, title_paragraph_1: Paragraph, title_paragraph_2: Paragraph,
                        text_paragraph: Paragraph) -> None:
        article = Article(
            id='article-4',
            language='en',
            paragraphs=[title_paragraph_1, title_paragraph_2, text_paragraph],
            topics=TopicList(topics=[Topic(id='topic', title='Topic')])
        )
        export_helper = ExportHelper(article)
        exporter = MarkdownExporter()

        lines = list(exporter._get_titles(export_helper))

        assert lines == [
            '# Main Title',
            '## Alternative Titles',
            '- **Alternative Title**',
            '',
        ]

    def test_get_speakers(self, title_paragraph_1: Paragraph, question_paragraph: Paragraph,
                          answer_paragraph: Paragraph) -> None:
        article = Article(id='article-6', language='en',
                          paragraphs=[title_paragraph_1, question_paragraph, answer_paragraph],
                          topics=TopicList(topics=[Topic(id='topic', title='Topic')])
                          )
        export_helper = ExportHelper(article)
        exporter = MarkdownExporter()

        lines = list(exporter._get_speakers(export_helper))

        assert lines[:3] == ['---', '### Speakers', '']
        assert lines[-1] == '---'
        assert set(lines[3:5]) == {'- spk_1', '- spk_2'}

    def test_get_qa_paragraphs(self, speaker_1: Speaker, speaker_2: Speaker, title_paragraph_1: Paragraph,
                               text_paragraph: Paragraph, question_paragraph: Paragraph,
                               answer_paragraph: Paragraph, topics: TopicList) -> None:
        segments = [
            SegmentSource(id='s0', text='segment-0', speaker_id=speaker_1.speaker_id),
            SegmentSource(id='s1', text='segment-1', speaker_id=speaker_2.speaker_id),
        ]
        transcript = Transcript(id='transcript-qa', language='en', segments=segments,
                                speakers=[speaker_1, speaker_2])
        article = Article(
            id='article-8',
            language='en',
            paragraphs=[title_paragraph_1, text_paragraph, question_paragraph, answer_paragraph],
            transcript=transcript,
            topics=topics,
        )
        export_helper = ExportHelper(article)
        exporter = MarkdownExporter()

        lines = list(exporter._get_qa_paragraphs(export_helper))

        assert lines == [
            '',
            '### Topic',
            '',
            'Intro',
            '',
            '**John Doe:** What about Acme Corp?',
            '',
            '**Max Mustermann:** "Answer about Acme Corp."',
            '',
        ]

    def test_include_metadata_adds_notes_and_links_occurrences(
            self,
            title_paragraph_1: Paragraph,
            question_paragraph: Paragraph,
            answer_paragraph: Paragraph,
            speaker_1: Speaker,
            speaker_2: Speaker,
            topics: TopicList,
    ) -> None:
        web_searches = WebSearchResultList(data=[
            WebSearchResult(
                id='acme-corp',
                text_occurrences=[
                    TextOccurrence(paragraph_id='p-answer-1', text_string='Acme Corp'),
                ],
                query='Acme Corp',
                summary='Acme Corp is a company.',
                sources=[WebSource(url='https://example.com', title='Example', snippet='Example snippet')],
            )
        ])
        article = Article(
            id='article',
            language='en',
            paragraphs=[
                title_paragraph_1,
                question_paragraph,
                answer_paragraph,
            ],
            transcript=Transcript(
                id='tr',
                language='en',
                segments=[
                    SegmentSource(id='s1', text='Question', speaker_id='spk_1'),
                    SegmentSource(id='s2', text='Answer', speaker_id='spk_2'),
                ],
                speakers=[speaker_1, speaker_2],
            ),
            topics=topics,
            metadata=ArticleMetadata(web_searches=web_searches),
        )
        exporter = MarkdownExporter()

        enriched_article = exporter.include_metadata(article)
        notes = list(exporter._get_notes())

        assert enriched_article.paragraphs[2].text == 'Answer about [Acme Corp](#ai-note-1).'
        assert notes[1] == '### Notes'
        assert '#### AI Note 1' in notes
        assert any('https://example.com' in line for line in notes)


class TestGoogleDocExporter:
    def test_get_titles(self, title_paragraph_1: Paragraph, title_paragraph_2: Paragraph,
                        text_paragraph: Paragraph) -> None:
        helper = ExportHelper(Article(
            id='article-9',
            language='en',
            paragraphs=[title_paragraph_1, title_paragraph_2, text_paragraph],
            topics=TopicList(topics=[Topic(id='topic', title='Topic')])
        ))
        exporter = GoogleDocExporter()

        lines = exporter._get_titles(helper)

        assert lines == ['Main Title',
                         'Alternative Titles',
                         '   -   Alternative Title',
                         '']
        assert exporter.titles == ('Main Title', ['Alternative Title'])

    def test_get_speakers(self, title_paragraph_1: Paragraph, question_paragraph: Paragraph,
                          answer_paragraph: Paragraph) -> None:
        article = Article(id='article-6', language='en',
                          paragraphs=[title_paragraph_1, question_paragraph, answer_paragraph],
                          topics=TopicList(topics=[Topic(id='topic', title='Topic')])
                          )
        export_helper = ExportHelper(article)
        exporter = GoogleDocExporter()

        lines = list(exporter._get_speakers(export_helper))

        assert lines[:2] == ['-------------------------------------------------------------------------------------',
                             'Speakers']
        assert lines[4:] == ['',
                             '-------------------------------------------------------------------------------------',
                             '']
        assert set(lines[2:4]) == {'   -   spk_1', '   -   spk_2'}

    def test_get_qa_paragraphs_formats_and_records_speakers(
            self,
            speaker_1: Speaker,
            speaker_2: Speaker,
            title_paragraph_1: Paragraph,
            text_paragraph: Paragraph,
            question_paragraph: Paragraph,
            answer_paragraph: Paragraph,
            topics: TopicList,
    ) -> None:
        segments = [
            SegmentSource(id='s0', text='segment-0', speaker_id=speaker_1.speaker_id),
            SegmentSource(id='s1', text='segment-1', speaker_id=speaker_2.speaker_id),
        ]
        transcript = Transcript(id='transcript-qa', language='en', segments=segments,
                                speakers=[speaker_1, speaker_2])
        article = Article(
            id='article-8',
            language='en',
            paragraphs=[title_paragraph_1, text_paragraph, question_paragraph, answer_paragraph],
            transcript=transcript,
            topics=topics,
        )
        exporter = GoogleDocExporter()
        export_helper = ExportHelper(article)

        lines = exporter._get_qa_paragraphs(export_helper)

        assert lines == [
            'Topic',
            'Intro',
            '',
            'John Doe: What about Acme Corp?',
            '',
            'Max Mustermann: "Answer about Acme Corp."',
            '',
        ]
        assert exporter.speakers == {'John Doe:', 'Max Mustermann:'}

    def test_set_formatting_applies_all_styles(
            self,
            speaker_1: Speaker,
            speaker_2: Speaker,
            title_paragraph_1: Paragraph,
            title_paragraph_2: Paragraph,
            text_paragraph: Paragraph,
            question_paragraph: Paragraph,
            answer_paragraph: Paragraph,
            topics: TopicList,
    ) -> None:
        transcript = Transcript(
            id='transcript-formatting',
            language='en',
            segments=[
                SegmentSource(id='s0', text='segment-0', speaker_id=speaker_1.speaker_id),
                SegmentSource(id='s1', text='segment-1', speaker_id=speaker_2.speaker_id),
            ],
            speakers=[speaker_1, speaker_2],
        )
        article = Article(
            id='article-formatting',
            language='en',
            paragraphs=[title_paragraph_1, title_paragraph_2, text_paragraph, question_paragraph, answer_paragraph],
            transcript=transcript,
            topics=topics,
        )
        helper = ExportHelper(article)
        exporter = GoogleDocExporter()
        # This sets the self.speakers and self.titles, which are necessary for the styling
        exporter._get_titles(helper)
        exporter._get_qa_paragraphs(helper)
        wrapper = Mock(spec_set=[
            'update_all_paragraphs_style_request',
            'update_paragraph_style_request',
            'update_text_style_request',
            'commit_requests',
        ])

        exporter._set_formatting(article, wrapper)

        wrapper.update_all_paragraphs_style_request.assert_called_once_with(line_spacing=1.15)
        paragraph_style_calls = wrapper.update_paragraph_style_request.call_args_list
        assert call(texts=['Main Title'], named_style='HEADING_1') in paragraph_style_calls
        assert call(texts=['Alternative Titles'], named_style='HEADING_2') in paragraph_style_calls
        assert call(texts=['Speakers'], named_style='HEADING_3') in paragraph_style_calls
        assert call(texts=['Summary'], named_style='HEADING_3') in paragraph_style_calls
        assert call(texts=['Topic'], named_style='HEADING_3') in paragraph_style_calls

        title_call, speakers_call = wrapper.update_text_style_request.call_args_list
        assert title_call == call(texts=['Alternative Title'], text_style='bold')

        speaker_args, speaker_kwargs = speakers_call
        assert not speaker_args
        assert speaker_kwargs['text_style'] == 'bold'
        assert speaker_kwargs['find_all']
        assert speaker_kwargs['skip_formatted_paragraphs'] is True
        assert set(speaker_kwargs['texts']) == {'John Doe:', 'Max Mustermann:'}

    def test_get_google_doc_paras_offset(self) -> None:
        wrapper = Mock()
        wrapper.get_document_paragraphs.return_value = [
            {
                'startIndex': 1,
                'endIndex': 5,
                'paragraph': {'elements': [{'textRun': {'content': 'Hello\n'}}]},
            },
            {
                'startIndex': 5,
                'endIndex': 10,
                'paragraph': {'elements': [{'textRun': {'content': 'World\n'}}]},
            },
        ]

        offsets = GoogleDocExporter.get_google_doc_paras_offset(wrapper)

        assert offsets == {'Hello': (1, 5), 'World': (5, 10)}

    def test_get_summary_paragraph_end_index(self) -> None:
        paragraph_by_id = {
            'p1': Paragraph(id='p1', type='summary', text='Summary', segment_refs=[]),
            'p2': Paragraph(id='p2', type='lead', text='Lead paragraph', segment_refs=[]),
        }
        offsets = {'Lead paragraph': (12, 20)}

        end_index = GoogleDocExporter._get_summary_paragraph_end_index(paragraph_by_id, offsets)

        assert end_index == 11

    @pytest.mark.parametrize(
        ('url', 'expected_url'),
        [
            ('https://youtu.be/NjpnRLBdkBo', 'https://youtu.be/NjpnRLBdkBo?t=12'),
            ('https://youtu.be/NjpnRLBdkBo?si=abc', 'https://youtu.be/NjpnRLBdkBo?si=abc&t=12'),
            ('https://www.youtube.com/watch?v=NjpnRLBdkBo', 'https://www.youtube.com/watch?v=NjpnRLBdkBo&t=12s'),
            ('https://www.youtube.com/watch?v=NjpnRLBdkBo&feature=share',
             'https://www.youtube.com/watch?v=NjpnRLBdkBo&feature=share&t=12s'),
            ('https://example.com/video', None)
        ],
    )
    def test_link_segment_timecodes(self, url: str, expected_url: str | None, topics: TopicList) -> None:
        article = Article(
            id='article-timecodes',
            language='en',
            paragraphs=[
                Paragraph(
                    id='p1',
                    type='text',
                    text='Intro',
                    segment_refs=[],
                    topic_id='topic',
                    source_timecodes=Timecodes(start_time=12.34, end_time=56.78),
                )
            ],
            topics=topics,
            transcript=Transcript(id='tr', language='en', segments=[], url=url),
        )
        wrapper = Mock()

        GoogleDocExporter.link_segment_timecodes(wrapper, article)

        if expected_url:
            wrapper.update_text_style_request.assert_called_once()
            _, kwargs = wrapper.update_text_style_request.call_args
            assert kwargs['url_link'] == expected_url
            assert kwargs['texts'] == ['[12.34, 56.78]']
            assert kwargs['find_all'] is True
            assert kwargs['skip_formatted_paragraphs'] is True
        else:
            wrapper.update_text_style_request.assert_not_called()
