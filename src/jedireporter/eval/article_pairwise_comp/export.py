from pathlib import Path
from typing import Literal

from google.oauth2.credentials import Credentials

from jedireporter.article import Article, Paragraph
from jedireporter.eval.article_pairwise_comp.models import ArticleComparisonResult, ArticlePair
from jedireporter.impex import GoogleDocExporter, MarkdownExporter
from jedireporter.transcript import Transcript
from jedireporter.utils import logging as logutil
from jedireporter.utils.gdoc import GoogleDocRequestsWrapper

LOG = logutil.getLogger(__package__, __file__)


class ComparisonMarkdownExporter:
    def __init__(self, output_dir: Path):
        self.output_dir: Path = output_dir

    @staticmethod
    def _article_to_markdown_lines(article: Article) -> list[str]:
        """Fixes article headings level to match the full comparison Markdown format."""
        raw_lines = list(MarkdownExporter().to_lines(article))
        converted: list[str] = []
        for line in raw_lines:
            if line.startswith('#'):
                heading_level = len(line) - len(line.lstrip('#'))
                rest = line[heading_level:]
                new_level = heading_level + 2
                converted.append('#' * new_level + rest)
            else:
                converted.append(line)
        return converted

    def write_markdown(self, pair: ArticlePair, result: ArticleComparisonResult) -> None:
        self.output_dir = self.output_dir

        lines: list[str] = [f'# {result.id}', '']
        winner_display = {
            'article_a': f'Article A ({pair.model_a_article[0]})',
            'article_b': f'Article B ({pair.model_b_article[0]})',
            'tie': 'Tie',
        }.get(result.winner, result.winner)
        lines.append(f'**Winner:** {winner_display}\\')
        if result.confidence is not None:
            lines.append(f'**Confidence:** {result.confidence:.2f}\\')
        top_justification = result.justification.replace('\n', '<br>')
        lines.append(f'**Justification:** {top_justification}')
        lines.append('')
        lines.append(f'| Criterion | Article A ({pair.model_a_article[0]}) | Article B ({pair.model_b_article[0]}) '
                     f'| Better | Justification |')
        lines.append('| --- | --- | --- | --- | --- |')
        criteria = result.criteria or []
        if not criteria:
            lines.append('| n/a | n/a | n/a | n/a | Judge did not return per-criterion assessments |')
        for criterion in criteria:
            name = criterion.name.replace('|', '\\|')
            justification = criterion.justification.replace('\n', '<br>').replace('|', '\\|')
            lines.append(
                f'| {criterion.id}. {name} '
                f'| {criterion.article_a_score.value} '
                f'| {criterion.article_b_score.value} '
                f'| {criterion.better.value} '
                f'| {justification} |'
            )
        lines.append('')
        lines.append(f'### Article A ({pair.model_a_article[0]})')
        lines.extend(self._article_to_markdown_lines(pair.model_a_article[1]))
        lines.append('')
        lines.append(f'### Article B ({pair.model_b_article[0]})')
        lines.extend(self._article_to_markdown_lines(pair.model_b_article[1]))
        lines.append('')

        markdown_path = self.output_dir / f'{result.id}.md'
        with markdown_path.open('w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))


class ComparisonGoogleDocExporter:
    def __init__(
            self,
            *,
            credentials: Credentials | None = None,
            service_credentials_path: Path | None = None,
            token_path: Path | str = 'token.json',
            request_wrapper: GoogleDocRequestsWrapper | None = None,
            google_folder: str | None = None
    ) -> None:
        self.google_folder: str | None = google_folder
        self.request_wrapper = request_wrapper or GoogleDocRequestsWrapper(
            credentials=credentials,
            client_token=token_path,
            service_credentials_path=service_credentials_path,
            batch_requests=True
        )
        self.article_exporters: dict[Literal['a', 'b'], GoogleDocExporter] = dict()

    def _add_summary_lines(self, result: ArticleComparisonResult) -> None:
        text = (f'{result.id}\n'
                f'Winner: {result.winner}\n'
                f'Confidence: {round(result.confidence, 2)}\n'
                f'Justification: {result.justification}\n')

        self.request_wrapper.insert_text_request(text)
        self.request_wrapper.update_paragraph_style_request(texts=[result.id], named_style='HEADING_1')
        names = ['Winner', 'Confidence', 'Justification']
        self.request_wrapper.update_text_style_request(texts=names, text_style='bold', skip_formatted_paragraphs=True)

    def _add_criteria_table(self, result: ArticleComparisonResult) -> None:
        # ------------------------------------------------------------
        # Criterion   | Article A | Article B | Better | Justification |
        # Criterion 1 |   fully   |  partly   |   A    |  lorem ipsum  |
        # ...
        row_nbr = len(result.criteria) + 1
        self.request_wrapper.insert_table_request(cols_nbr=5, rows_nbr=row_nbr)

    @staticmethod
    def _get_criterion_table_header(pair: ArticlePair) -> list[str]:
        return ['Criterion',
                f'Article A [{pair.model_a_article[0]}]',
                f'Article B [{pair.model_b_article[0]}]',
                'Better',
                'Justification']

    def _fill_criteria_table_rows(self, pair: ArticlePair, result: ArticleComparisonResult) -> None:
        header = self._get_criterion_table_header(pair)
        for col_idx, header_cell in enumerate(header):
            self.request_wrapper.fill_table_cell_value_request(0, 0, col_idx, header_cell)
        if criteria := result.criteria:
            for criterion_idx, criterion in enumerate(criteria, start=1):
                row_values = [f'{criterion.id}. {criterion.name}', criterion.article_a_score.value,
                              criterion.article_b_score.value, criterion.better.value, criterion.justification]
                for col_idx, cell_value in enumerate(row_values):
                    self.request_wrapper.fill_table_cell_value_request(0, criterion_idx, col_idx, cell_value)
        else:
            for col_idx, cell_text in enumerate(
                    ['n/a', 'n/a', 'n/a', 'n/a', 'Judge did not return per-criterion assessments']):
                self.request_wrapper.fill_table_cell_value_request(0, 1, col_idx, cell_text)

    def _make_criterion_table_header_bold(self, pair: ArticlePair) -> None:
        header = self._get_criterion_table_header(pair)
        table_doc_start_index = self.request_wrapper.get_ith_table_doc_range(0)[0]
        self.request_wrapper.update_text_style_request(
            texts=header, text_style='bold', search_index_start=table_doc_start_index, skip_formatted_paragraphs=True)

    def _article_columns(self, pair: ArticlePair) -> tuple[str, str]:
        google_exporter_a = GoogleDocExporter()
        article_a_text = google_exporter_a.article_to_text(pair.model_a_article[1])
        self.article_exporters['a'] = google_exporter_a
        google_exporter_b = GoogleDocExporter()
        article_b_text = google_exporter_b.article_to_text(pair.model_b_article[1])
        self.article_exporters['b'] = google_exporter_b
        column_a = f'Article A ({pair.model_a_article[0]})\n{article_a_text}'
        column_b = f'Article B ({pair.model_b_article[0]})\n{article_b_text}'
        return column_a, column_b

    def _set_main_article_titles_style(self, pair: ArticlePair) -> None:
        article_a_model = f'Article A ({pair.model_a_article[0]})'
        article_b_model = f'Article B ({pair.model_b_article[0]})'
        article_a_title = pair.model_a_article[1].paragraphs[0].text
        article_b_title = pair.model_b_article[1].paragraphs[0].text
        articles_table_start_index = self.request_wrapper.get_ith_table_doc_range(table_idx=1)[0]
        main_titles = [article_a_model, article_b_model, article_a_title, article_b_title]
        self.request_wrapper.update_paragraph_style_request(
            texts=main_titles, named_style='HEADING_3', search_index_start=articles_table_start_index,
            skip_formatted_paragraphs=True)

    def _set_alternative_titles_style(self, pair: ArticlePair) -> None:
        article_a_table_cell_index = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=0)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Alternative Titles'], named_style='HEADING_4', search_index_start=article_a_table_cell_index,
            skip_formatted_paragraphs=True)
        article_b_table_cell_index = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=1)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Alternative Titles'], named_style='HEADING_4', search_index_start=article_b_table_cell_index,
            skip_formatted_paragraphs=True)

        article_a_alternative_titles = [paragraph.text for paragraph in pair.model_a_article[1].paragraphs if
                                        paragraph.type == 'title'][1:]
        article_b_alternative_titles = [paragraph.text for paragraph in pair.model_b_article[1].paragraphs if
                                        paragraph.type == 'title'][1:]
        alternative_titles = article_a_alternative_titles + article_b_alternative_titles
        self.request_wrapper.update_text_style_request(
            texts=alternative_titles, text_style='bold', search_index_start=article_a_table_cell_index,
            skip_formatted_paragraphs=True)

    def _set_speakers_and_summary_section_style(self) -> None:
        article_a_cell_idx = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=0)
        article_b_cell_idx = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=1)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Speakers'], named_style='HEADING_4', search_index_start=article_a_cell_idx,
            skip_formatted_paragraphs=True)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Speakers'], named_style='HEADING_4', search_index_start=article_b_cell_idx,
            skip_formatted_paragraphs=True)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Summary'], named_style='HEADING_4', search_index_start=article_a_cell_idx,
            skip_formatted_paragraphs=True)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Summary'], named_style='HEADING_4', search_index_start=article_b_cell_idx,
            skip_formatted_paragraphs=True)

    def _add_links(self, doc_exporter: GoogleDocExporter, article: Article,
                   search_index_start: int | None = None) -> None:
        if article.metadata and article.metadata.web_searches and article.metadata.web_searches.data:
            heading_ids = self.request_wrapper.get_heading_ids_by_titles(doc_exporter.note_headings,
                                                                         search_index_start=search_index_start)
            google_doc_paras_offset = doc_exporter.get_google_doc_paras_offset(self.request_wrapper)
            paragraph_by_id: dict[str, Paragraph] = {p.id: p for p in article.paragraphs}
            for web_info_id, web_info in enumerate(article.metadata.web_searches.data):
                heading_id = heading_ids[web_info_id]
                if not heading_id:
                    LOG.warning(f'Heading ID missing for AI Note {web_info_id + 1}, link was not added.')
                    continue
                for text in web_info.text_occurrences:
                    doc_exporter.add_links_to_text_occurrence(
                        text, heading_id, paragraph_by_id, google_doc_paras_offset, self.request_wrapper)

        for source in doc_exporter.sources:
            self.request_wrapper.update_text_style_request(
                texts=[source[0]],
                url_link=source[1],
                find_all=True,
                skip_formatted_paragraphs=True,
                search_index_start=search_index_start
            )
        # Link transcript timecodes (e.g. YouTube) when available.
        doc_exporter.link_segment_timecodes(self.request_wrapper, article)

    def _set_notes_section_style(self, pair: ArticlePair) -> None:
        article_a_cell_idx = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=0)
        article_b_cell_idx = self.request_wrapper.get_cell_index_from_table(table_idx=1, row=0, col=1)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Notes'], named_style='HEADING_4', search_index_start=article_a_cell_idx,
            skip_formatted_paragraphs=True)
        self.request_wrapper.update_paragraph_style_request(
            texts=['Notes'], named_style='HEADING_4', search_index_start=article_b_cell_idx,
            skip_formatted_paragraphs=True)
        self.request_wrapper.update_paragraph_style_request(
            texts=self.article_exporters['a'].note_headings, named_style='HEADING_5',
            search_index_start=article_a_cell_idx)
        self.request_wrapper.update_paragraph_style_request(
            texts=self.article_exporters['b'].note_headings, named_style='HEADING_5',
            search_index_start=article_b_cell_idx)
        self.request_wrapper.commit_requests()

        self._add_links(self.article_exporters['a'], pair.model_a_article[1], search_index_start=article_a_cell_idx)
        self._add_links(self.article_exporters['b'], pair.model_b_article[1], search_index_start=article_b_cell_idx)

    @staticmethod
    def _get_speaker_set_for_article(article: Article) -> set[str]:
        transcript = Transcript.model_validate(article.transcript)
        if transcript_speakers := transcript.speakers:
            return {f'{speaker.name}:' if speaker.name else f'{speaker.speaker_id}:' for speaker in transcript_speakers}
        else:
            return {f'{paragraph.speaker}:' for paragraph in article.paragraphs if paragraph.speaker}

    def _make_all_qa_section_speakers_bold(self, pair: ArticlePair) -> None:
        index_start = self.request_wrapper.get_ith_table_doc_range(table_idx=1)[0]
        speakers_a = self._get_speaker_set_for_article(pair.model_a_article[1])
        speakers_b = self._get_speaker_set_for_article(pair.model_b_article[1])
        speakers = list(speakers_a | speakers_b)
        self.request_wrapper.update_text_style_request(
            texts=speakers, text_style='bold', search_index_start=index_start, find_all=True,
            skip_formatted_paragraphs=True)

    def _set_topic_titles_style(self) -> None:
        self.request_wrapper.update_paragraph_style_request(texts=list(self.article_exporters['a'].topic_headings),
                                                            named_style='HEADING_4')
        self.request_wrapper.update_paragraph_style_request(texts=list(self.article_exporters['b'].topic_headings),
                                                            named_style='HEADING_4')

    def write_to_google_doc(self, pair: ArticlePair, result: ArticleComparisonResult) -> str | None:
        # --- Reset article exporters ---
        self.article_exporters = dict()
        # --- Create doc ---
        self.request_wrapper.create_document_request(pair.model_a_article[1].id, flip_orientation=True,
                                                     parent_folder_id=self.google_folder)
        # --- Add summary ---
        self._add_summary_lines(result)
        # --- Add criteria table ---
        self._add_criteria_table(result)
        column_widths_inches = [0.98, 1.01, 1.01, 0.75, 5.3]
        self.request_wrapper.update_table_column_properties_request(0, column_widths_inches)
        self._fill_criteria_table_rows(pair, result)
        self._make_criterion_table_header_bold(pair)
        #  --- Add articles ---
        # Create a non-border table to have the articles side-by-side
        self.request_wrapper.insert_table_request(cols_nbr=2, rows_nbr=1)
        column_a, column_b = self._article_columns(pair)
        self.request_wrapper.fill_table_cell_value_request(table_idx=1, row=0, col=0, text=column_a)
        self.request_wrapper.fill_table_cell_value_request(table_idx=1, row=0, col=1, text=column_b)

        self._set_main_article_titles_style(pair)
        self._set_alternative_titles_style(pair)
        self._set_speakers_and_summary_section_style()
        self._set_topic_titles_style()

        self.request_wrapper.update_table_style_request(table_idx=1, line_spacing=1.15)
        self.request_wrapper.update_table_cell_style_request(table_idx=1, remove_border=True)
        # Split heading styling and bold speakers in Q&A part
        self.request_wrapper.commit_requests()
        self._make_all_qa_section_speakers_bold(pair)
        self._set_notes_section_style(pair)
        self.request_wrapper.commit_requests()
