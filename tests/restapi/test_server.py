from unittest.mock import MagicMock

from starlette.testclient import TestClient

from jedireporter.article import Article, Paragraph, SegmentRef, Topic, TopicList
from jedireporter.restapi.server import _VER, Message
from jedireporter.transcript import Segment, Speaker, Transcript
from jedireporter.workflow import InterviewProcessor, State


class TestServer:
    def test_get_status(self, client: TestClient):
        resp = client.get('/status')
        assert resp.status_code == 200
        resp_obj = Message.model_validate_json(resp.text)
        assert isinstance(resp_obj, Message)
        assert resp_obj == Message(message=f'JEDI Reporter Service {_VER} - Ready.')


class TestGenerateArticle:
    def test_generate_article_success(self, client: TestClient):
        # 1. Create test input (APITranscript as dict)
        input_transcript = {
            "id": "test-transcript-1",
            "language": "en",
            "speakers": [
                {"speakerId": "host", "role": "host", "name": "Jane Host"},
                {"speakerId": "guest", "role": "guest", "name": "John Guest"},
            ],
            "segments": [
                {"id": "seg-1", "speakerId": "host", "text": "Welcome to the show. What is your project about?"},
                {
                    "id": "seg-2",
                    "speakerId": "guest",
                    "text": "Thanks for having me. Our project focuses on AI safety."
                },
                {"id": "seg-3", "speakerId": "host", "text": "That sounds fascinating. Tell us more."},
                {
                    "id": "seg-4",
                    "speakerId": "guest",
                    "text": "We research alignment techniques to ensure AI systems are safe."
                },
            ],
        }

        # 2. Create mock return values for each subgraph
        mock_fixed_transcript = Transcript(
            id="test-transcript-1",
            language="en",
            speakers=[
                Speaker(speaker_id="host", role="host", name="Jane Host"),
                Speaker(speaker_id="guest", role="guest", name="John Guest"),
            ],
            segments=[
                Segment(id="seg-1", speaker_id="host", text="Welcome to the show. What is your project about?",
                        segment_references={"seg-1"}),
                Segment(id="seg-2", speaker_id="guest",
                        text="Thanks for having me. Our project focuses on AI safety.",
                        segment_references={"seg-2"}),
                Segment(id="seg-3", speaker_id="host", text="That sounds fascinating. Tell us more.",
                        segment_references={"seg-3"}),
                Segment(id="seg-4", speaker_id="guest",
                        text="We research alignment techniques to ensure AI systems are safe.",
                        segment_references={"seg-4"}),
            ],
        )

        mock_article = Article(
            id="article-test-transcript-1",
            language="en",
            topics=TopicList(topics=[
                Topic(id="ai-safety", title="AI Safety"),
            ]),
            paragraphs=[
                Paragraph(
                    id="p-1",
                    type="title",
                    text="AI Safety Research: An Interview",
                    segment_refs=[SegmentRef(segment_id="seg-1")],
                ),
                Paragraph(
                    id="p-2",
                    type="question",
                    text="What is your project about?",
                    speaker="host",
                    topic_id="ai-safety",
                    segment_refs=[SegmentRef(segment_id="seg-1")],
                ),
                Paragraph(
                    id="p-3",
                    type="answer",
                    text="Our project focuses on AI safety. We research alignment techniques "
                         "to ensure AI systems are safe.",
                    speaker="guest",
                    topic_id="ai-safety",
                    segment_refs=[SegmentRef(segment_id="seg-2"), SegmentRef(segment_id="seg-4")],
                ),
            ],
        )

        # 3. Create mock functions that preserve their names for the workflow graph
        # The StateGraph uses __name__ for node naming, so we need to set it explicitly
        def run_fix_transcript_subgraph(state: State) -> dict:
            return {'fixed': mock_fixed_transcript}

        def run_create_article_subgraph(state: State) -> dict:
            return {'article': mock_article}

        def modify_style(state: State) -> dict:
            return {'styled': mock_article}

        # Store original methods
        original_fix = InterviewProcessor.run_fix_transcript_subgraph
        original_create = InterviewProcessor.run_create_article_subgraph
        original_style = InterviewProcessor.modify_style

        # Wrap mock functions with MagicMock for call tracking while preserving __name__
        mock_fix = MagicMock(side_effect=run_fix_transcript_subgraph)
        mock_fix.__name__ = 'run_fix_transcript_subgraph'
        mock_create = MagicMock(side_effect=run_create_article_subgraph)
        mock_create.__name__ = 'run_create_article_subgraph'
        mock_style = MagicMock(side_effect=modify_style)
        mock_style.__name__ = 'modify_style'

        try:
            # Patch methods
            InterviewProcessor.run_fix_transcript_subgraph = mock_fix
            InterviewProcessor.run_create_article_subgraph = mock_create
            InterviewProcessor.modify_style = mock_style

            # 4. POST to /v1/generate-article
            response = client.post('/v1/generate-article', json=input_transcript)

            # 5. Assert response status and structure
            assert response.status_code == 200, f"Response: {response.json()}"

            result = response.json()
            assert result["id"] == "article-test-transcript-1"
            assert result["language"] == "en"
            assert len(result["paragraphs"]) == 3

            # Check paragraph types
            assert result["paragraphs"][0]["type"] == "title"
            assert result["paragraphs"][1]["type"] == "question"
            assert result["paragraphs"][2]["type"] == "answer"

            # Check that transcript is included for traceability
            assert "transcript" in result
            assert result["transcript"]["id"] == "test-transcript-1"
            assert len(result["transcript"]["segments"]) == 4
            assert len(result["transcript"]["speakers"]) == 2

            # Verify mocked methods were called
            mock_fix.assert_called_once()
            mock_create.assert_called_once()
            mock_style.assert_called_once()
        finally:
            # Restore original methods
            InterviewProcessor.run_fix_transcript_subgraph = original_fix
            InterviewProcessor.run_create_article_subgraph = original_create
            InterviewProcessor.modify_style = original_style
