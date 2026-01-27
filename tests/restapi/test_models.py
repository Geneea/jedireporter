"""Tests for REST API models and conversions."""

import pytest
from pydantic import AnyUrl

from jedireporter.article import (
    Article,
    ArticleMetadata,
    Paragraph,
    SegmentRef,
    TextOccurrence,
    Topic,
    TopicList,
    WebSearchResult,
    WebSearchResultList,
    WebSource,
)
from jedireporter.restapi.models import (
    APIArticle, APIParagraph, APISegment, APISegmentRef, APISpeaker,
    APITimecodes, APITranscript)
from jedireporter.transcript import Segment, SegmentSource, Speaker, Timecodes, Transcript


class TestAPITimecodes:
    """Tests for APITimecodes model."""

    def test_to_internal(self):
        """Test conversion from API model to internal model."""
        api_tc = APITimecodes(start_time=0.0, end_time=10.5)
        internal_tc = api_tc.to_internal()

        assert isinstance(internal_tc, Timecodes)
        assert internal_tc.start_time == 0.0
        assert internal_tc.end_time == 10.5

    def test_from_internal(self):
        """Test conversion from internal model to API model."""
        internal_tc = Timecodes(start_time=5.5, end_time=15.3)
        api_tc = APITimecodes.from_internal(internal_tc)

        assert isinstance(api_tc, APITimecodes)
        assert api_tc.start_time == 5.5
        assert api_tc.end_time == 15.3

    def test_round_trip(self):
        """Test round-trip conversion maintains data integrity."""
        original = APITimecodes(start_time=1.0, end_time=2.0)
        round_trip = APITimecodes.from_internal(original.to_internal())

        assert round_trip.start_time == original.start_time
        assert round_trip.end_time == original.end_time

    def test_serialization_camel_case(self):
        """Test that serialization uses camelCase."""
        api_tc = APITimecodes(start_time=0.0, end_time=10.0)
        json_dict = api_tc.model_dump()

        assert 'startTime' in json_dict
        assert 'endTime' in json_dict
        assert json_dict['startTime'] == 0.0
        assert json_dict['endTime'] == 10.0

    def test_deserialization_snake_case(self):
        """Test that deserialization accepts snake_case."""
        data = {'start_time': 1.5, 'end_time': 3.5}
        api_tc = APITimecodes.model_validate(data)

        assert api_tc.start_time == 1.5
        assert api_tc.end_time == 3.5

    def test_deserialization_camel_case(self):
        """Test that deserialization accepts camelCase."""
        data = {'startTime': 2.5, 'endTime': 4.5}
        api_tc = APITimecodes.model_validate(data)

        assert api_tc.start_time == 2.5
        assert api_tc.end_time == 4.5


class TestAPISpeaker:
    """Tests for APISpeaker model."""

    def test_to_internal(self):
        """Test conversion from API model to internal model."""
        api_speaker = APISpeaker(
            speaker_id='spk_0',
            role='host',
            name='Jane Doe',
            description='Lead interviewer'
        )
        internal_speaker = api_speaker.to_internal()

        assert isinstance(internal_speaker, Speaker)
        assert internal_speaker.speaker_id == 'spk_0'
        assert internal_speaker.role == 'host'
        assert internal_speaker.name == 'Jane Doe'
        assert internal_speaker.description == 'Lead interviewer'

    def test_from_internal(self):
        """Test conversion from internal model to API model."""
        internal_speaker = Speaker(
            speaker_id='guest_1',
            role='guest',
            name='John Smith'
        )
        api_speaker = APISpeaker.from_internal(internal_speaker)

        assert isinstance(api_speaker, APISpeaker)
        assert api_speaker.speaker_id == 'guest_1'
        assert api_speaker.role == 'guest'
        assert api_speaker.name == 'John Smith'
        assert api_speaker.description is None

    def test_round_trip(self):
        """Test round-trip conversion maintains data integrity."""
        original = APISpeaker(speaker_id='spk_1', role='other')
        round_trip = APISpeaker.from_internal(original.to_internal())

        assert round_trip.speaker_id == original.speaker_id
        assert round_trip.role == original.role
        assert round_trip.name == original.name
        assert round_trip.description == original.description


class TestAPISegment:
    """Tests for APISegment model."""

    def test_to_internal_without_timecodes(self):
        """Test conversion without timecodes."""
        api_segment = APISegment(
            id='seg-1',
            text='Hello world',
            speaker_id='spk_0'
        )
        internal_segment = api_segment.to_internal()

        assert isinstance(internal_segment, SegmentSource)
        assert internal_segment.id == 'seg-1'
        assert internal_segment.text == 'Hello world'
        assert internal_segment.speaker_id == 'spk_0'
        assert internal_segment.timecodes is None

    def test_to_internal_with_timecodes(self):
        """Test conversion with timecodes."""
        api_segment = APISegment(
            id='seg-2',
            text='Test segment',
            speaker_id='spk_1',
            timecodes=APITimecodes(start_time=0.0, end_time=5.0)
        )
        internal_segment = api_segment.to_internal()

        assert isinstance(internal_segment, SegmentSource)
        assert internal_segment.timecodes is not None
        assert internal_segment.timecodes.start_time == 0.0
        assert internal_segment.timecodes.end_time == 5.0

    def test_from_internal_excludes_segment_references(self):
        """Test that segment_references is excluded in API conversion."""
        internal_segment = Segment(
            id='seg-3',
            text='Sample text',
            speaker_id='spk_0',
            segment_references={'orig-1', 'orig-2'}  # Internal field
        )
        api_segment = APISegment.from_internal(internal_segment)

        # API segment should not expose segment_references
        assert not hasattr(api_segment, 'segment_references')

    def test_serialization_camel_case(self):
        """Test that serialization uses camelCase."""
        api_segment = APISegment(
            id='seg-1',
            text='Test',
            speaker_id='spk_0'
        )
        json_dict = api_segment.model_dump()

        assert 'speakerId' in json_dict
        assert 'speaker_id' not in json_dict
        assert json_dict['speakerId'] == 'spk_0'


class TestApiTranscript:
    def test_to_internal_minimal(self):
        """Test conversion with minimal required fields."""
        api_transcript = APITranscript(
            id='test-1',
            language='en',
            segments=[
                APISegment(id='seg-1', text='Hello', speaker_id='spk_0')
            ]
        )
        internal_transcript = api_transcript.to_internal()

        assert isinstance(internal_transcript, Transcript)
        assert internal_transcript.id == 'test-1'
        assert internal_transcript.language == 'en'
        assert len(internal_transcript.segments) == 1
        assert internal_transcript.speakers is None
        assert internal_transcript.url is None

    def test_to_internal_complete(self):
        """Test conversion with all fields."""
        api_transcript = APITranscript(
            id='test-2',
            language='cs',
            segments=[
                APISegment(
                    id='seg-1',
                    text='Dobrý den',
                    speaker_id='host',
                    timecodes=APITimecodes(start_time=0.0, end_time=2.0)
                )
            ],
            speakers=[
                APISpeaker(speaker_id='host', role='host', name='Moderator')
            ],
            url=AnyUrl('https://example.com/interview.mp3')
        )
        internal_transcript = api_transcript.to_internal()

        assert isinstance(internal_transcript, Transcript)
        assert len(internal_transcript.segments) == 1
        assert internal_transcript.segments[0].timecodes is not None
        assert len(internal_transcript.speakers) == 1
        assert str(internal_transcript.url) == 'https://example.com/interview.mp3'

    def test_serialization_camel_case(self):
        """Test that serialization uses camelCase."""
        api_transcript = APITranscript(
            id='test-1',
            language='en',
            segments=[APISegment(id='seg-1', text='Test', speaker_id='spk_0')]
        )
        json_dict = api_transcript.model_dump()

        assert 'speaker_id' not in str(json_dict)
        # Check nested segment uses camelCase
        assert 'speakerId' in json_dict['segments'][0]


class TestAPISegmentRef:
    """Tests for APISegmentRef model."""

    def test_from_internal_complete(self):
        """Test conversion with all fields."""
        internal_ref = SegmentRef(segment_id='seg-1', offset=5, length=10)
        api_ref = APISegmentRef.from_internal(internal_ref)

        assert isinstance(api_ref, APISegmentRef)
        assert api_ref.segment_id == 'seg-1'
        assert api_ref.offset == 5
        assert api_ref.length == 10

    def test_from_internal_minimal(self):
        """Test conversion with minimal fields."""
        internal_ref = SegmentRef(segment_id='seg-2')
        api_ref = APISegmentRef.from_internal(internal_ref)

        assert api_ref.segment_id == 'seg-2'
        assert api_ref.offset is None
        assert api_ref.length is None


class TestAPIParagraph:
    """Tests for APIParagraph model."""

    def test_from_internal_title(self):
        """Test conversion of title paragraph."""
        internal_para = Paragraph(
            id='p01-title',
            type='title',
            text='Interview Title',
            segment_refs=[SegmentRef(segment_id='seg-1')]
        )
        api_para = APIParagraph.from_internal(internal_para)

        assert isinstance(api_para, APIParagraph)
        assert api_para.id == 'p01-title'
        assert api_para.type == 'title'
        assert api_para.text == 'Interview Title'
        assert api_para.speaker is None
        assert len(api_para.segment_refs) == 1

    def test_from_internal_question(self):
        """Test conversion of question paragraph."""
        internal_para = Paragraph(
            id='p02-question',
            type='question',
            text='What do you think?',
            speaker='host',
            segment_refs=[SegmentRef(segment_id='seg-2', offset=0, length=10)],
            topic_id='topic-1'
        )
        api_para = APIParagraph.from_internal(internal_para)

        assert api_para.id == 'p02-question'
        assert api_para.type == 'question'
        assert api_para.speaker == 'host'
        assert api_para.segment_refs[0].offset == 0
        assert api_para.segment_refs[0].length == 10


class TestApiArticle:
    def test_from_internal_raises_without_transcript(self):
        """Test that conversion fails if article.transcript is None."""
        article = Article(
            id='article-1',
            language='en',
            paragraphs=[],
            transcript=None,
            topics=TopicList(topics=[Topic(id='topic-1', title='Topic')])
        )

        with pytest.raises(ValueError, match='Article must have transcript set'):
            APIArticle.from_internal(article)

    def test_from_internal_complete(self):
        """Test conversion with complete article."""
        transcript = Transcript(
            id='test-1',
            language='en',
            segments=[
                Segment(id='seg-1', text='Hello', speaker_id='spk_0', segment_references={'orig-1'},
                        timecodes=Timecodes(start_time=0.0, end_time=1.0))
            ],
            speakers=[
                Speaker(speaker_id='spk_0', role='host', name='Host')
            ]
        )

        article = Article(
            id='article-test-1',
            language='en',
            paragraphs=[
                Paragraph(
                    id='p01-title',
                    type='title',
                    text='Test Article',
                    segment_refs=[SegmentRef(segment_id='seg-1')],
                    source_timecodes=Timecodes(start_time=0.0, end_time=1.0)
                )
            ],
            transcript=transcript,
            topics=TopicList(topics=[Topic(id='topic-1', title='Topic')])
        )

        api_output = APIArticle.from_internal(article)

        assert isinstance(api_output, APIArticle)
        assert api_output.id == 'article-test-1'
        assert api_output.language == 'en'
        assert len(api_output.paragraphs) == 1
        assert api_output.paragraphs[0].type == 'title'
        assert api_output.paragraphs[0].source_timecodes.start_time == 0.0

        # Verify transcript is included
        assert isinstance(api_output.transcript, APITranscript)
        assert api_output.transcript.id == 'test-1'
        assert len(api_output.transcript.segments) == 1
        assert len(api_output.transcript.speakers) == 1

    def test_from_internal_excludes_debug_info(self):
        """Test that debug_info is excluded from API output."""
        transcript = Transcript(
            id='test-1',
            language='en',
            segments=[Segment(id='seg-1', text='Test', speaker_id='spk_0', segment_references={'orig-1'})]
        )

        article = Article(
            id='article-1',
            language='en',
            paragraphs=[],
            topics=TopicList(topics=[Topic(id='topic-1', title='Topic')]),
            transcript=transcript,
            debug_info={'internal': 'data'}  # Internal field
        )

        api_output = APIArticle.from_internal(article)

        # API output should not expose debug_info
        assert not hasattr(api_output, 'debug_info')

    def test_serialization_includes_transcript(self):
        """Test that serialization includes the full transcript."""
        transcript = Transcript(
            id='test-1',
            language='en',
            segments=[Segment(id='seg-1', text='Test', speaker_id='spk_0', segment_references={'orig-1'})]
        )

        article = Article(
            id='article-1',
            language='en',
            paragraphs=[
                Paragraph(
                    id='p01-content',
                    type='text',
                    text='Content',
                    segment_refs=[SegmentRef(segment_id='seg-1')],
                    topic_id='topic-1'
                )
            ],
            transcript=transcript,
            topics=TopicList(topics=[Topic(id='topic-1', title='Topic')])
        )

        api_output = APIArticle.from_internal(article)
        json_dict = api_output.model_dump()

        assert 'transcript' in json_dict
        assert json_dict['transcript']['id'] == 'test-1'
        assert len(json_dict['transcript']['segments']) == 1
        # Verify camelCase in nested structures
        assert 'speakerId' in json_dict['transcript']['segments'][0]

    def test_from_internal_includes_metadata(self):
        """Test that web metadata is included when present."""
        transcript = Transcript(
            id='test-1',
            language='en',
            segments=[Segment(id='seg-1', text='Test', speaker_id='spk_0', segment_references={'orig-1'})]
        )
        web_searches = WebSearchResultList(
            data=[
                WebSearchResult(
                    id='subject-1',
                    text_occurrences=[TextOccurrence(paragraph_id='p01-title', text_string='Example Org')],
                    query='Example Org company profile',
                    summary='Example Org is a fictional company used for documentation.',
                    sources=[WebSource(url='https://example.com', title='Example Source', snippet='Example snippet')],
                )
            ]
        )
        article = Article(
            id='article-1',
            language='en',
            paragraphs=[
                Paragraph(
                    id='p01-title',
                    type='title',
                    text='Example Org interview',
                    segment_refs=[SegmentRef(segment_id='seg-1')]
                )
            ],
            topics=TopicList(topics=[Topic(id='topic-1', title='Topic')]),
            transcript=transcript,
            metadata=ArticleMetadata(web_searches=web_searches),
        )

        api_output = APIArticle.from_internal(article)

        assert api_output.metadata is not None
        assert api_output.metadata.web_searches is not None
        assert api_output.metadata.web_searches[0].id == 'subject-1'
