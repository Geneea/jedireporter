from importlib.resources import read_text

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from jedireporter.article import Article, Paragraph, TopicList
from jedireporter.llm import InstructorLLM
from jedireporter.transcript import Transcript


class CreateArticleState(BaseModel):
    """
    Pydantic state for the create-article subgraph.
    Contains the fixed transcript as input and produces the article as output.
    """
    fixed: Transcript
    detected_topics: TopicList | None = None
    topics: TopicList | None = None  # selected topics used for article creation
    article: Article | None = None


class CreateArticleNodes:
    def __init__(self, llm: InstructorLLM) -> None:
        self.llm = llm

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompt = read_text(f'{__package__}.resources.templates.create_article', f'{name}.txt')
        return prompt

    @staticmethod
    def _topic_scope_instruction(topics: TopicList) -> str:
        selected = ', '.join(topic.id for topic in topics.topics)
        return f'Only include Q&A and narrative that belong to the selected topics ({selected}).'

    def detect_topics(self, state: CreateArticleState) -> dict[str, TopicList]:
        if state.fixed is None:
            raise ValueError('Fixed transcript not available before detect_topics step')
        prompt = self._load_prompt('find_topics').format(source=state.fixed.model_dump_json())
        topics: TopicList = self.llm.get_completion(prompt, structured_output_type=TopicList)
        return {'detected_topics': topics}

    @staticmethod
    def select_topics(state: CreateArticleState) -> dict[str, TopicList]:
        if state.detected_topics is None:
            raise ValueError('Detected topics not available before select_topics step')
        detected = state.detected_topics.topics
        if not detected:
            raise ValueError('No topics to select from')
        # Use all detected topics for article generation
        return {'topics': TopicList(topics=list(detected))}

    @staticmethod
    def _fill_paragraph_ids(article: Article) -> None:
        for i, paragraph in enumerate(article.paragraphs):
            paragraph.id = f'par_{i}'

    # Single node for now; can be expanded later with more nodes
    def create_article(self, state: CreateArticleState) -> dict[str, Article]:
        if state.topics is None:
            raise ValueError('No topics available')
        if state.fixed is None:
            raise ValueError('Fixed transcript not available before create_article step')

        topics = state.topics
        if not topics.topics:
            raise ValueError('No topics available')
        topics_json_str = ', '.join(t.model_dump_json() for t in topics.topics)
        topic_scope_instruction = self._topic_scope_instruction(topics)
        prompt = self._load_prompt('create_article').format(
            source=state.fixed.model_dump_json(),
            topics=topics_json_str,
            topic_scope=topic_scope_instruction,
        )
        article = self.llm.get_completion(prompt, structured_output_type=Article)

        article.topics = topics
        return {'article': article}

    def generate_summary(self, state: CreateArticleState) -> dict[str, Article]:
        """
        Generate a concise summary paragraph (type='summary') from the already created article.
        The summary consists of 3-5 key bullet points depending on article length.
        """
        if state.article is None:
            raise ValueError('Article not available before generate_summary step')

        prompt = self._load_prompt('summarize_article').format(article=state.article.model_dump_json())
        summary_par = self.llm.get_completion(prompt, structured_output_type=Paragraph)

        # Ensure correct type
        if summary_par.type != 'summary':
            summary_par.type = 'summary'  # force the expected type
        # Insert summary after the last title paragraph
        insert_pos = 0
        for i, paragraph in enumerate(state.article.paragraphs):
            if paragraph.type == 'title':
                insert_pos = i + 1
        state.article.paragraphs.insert(insert_pos, summary_par)
        self._fill_paragraph_ids(state.article)

        return {'article': state.article}

    def build_subgraph(self) -> CompiledStateGraph:
        builder = StateGraph(CreateArticleState)
        builder.add_node('detect_topics', self.detect_topics)
        builder.add_node('select_topics', self.select_topics)
        builder.add_node('create_article', self.create_article)
        builder.add_node('generate_summary', self.generate_summary)
        builder.add_edge(START, 'detect_topics')
        builder.add_edge('detect_topics', 'select_topics')
        builder.add_edge('select_topics', 'create_article')
        builder.add_edge('create_article', 'generate_summary')
        builder.add_edge('generate_summary', END)
        return builder.compile()
