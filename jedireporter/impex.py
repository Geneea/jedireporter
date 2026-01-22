import argparse
import glob
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from jedireporter.article import Article, TextOccurrence, WebSearchResult
from jedireporter.article import Paragraph, Topic
from jedireporter.transcript import SegmentSource, Speaker, Transcript
from jedireporter.utils import cli_utils as cliutil
from jedireporter.utils import dict_utils as dictutil
from jedireporter.utils import logging as logutil
from jedireporter.utils.gdoc import GoogleDocRequestsWrapper

LOG = logutil.getLogger(__package__, __file__)


class PlainTextImporter:
    _RE_SPEAKER_TEXT = re.compile(r'^([-_\w\s]+):\s*(.*)$')

    def __init__(self, *, transcript_id: str, language: str | None = None, url: str | None = None):
        self._transcript_id = transcript_id
        self._language = language
        self._url = url

    def from_lines(self, lines: Iterable[str]) -> Transcript:
        segments: list[SegmentSource] = []
        speakers: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue  # silently ignore empty lines

            m = self._RE_SPEAKER_TEXT.match(line)
            if not m:
                LOG.warning(f'Invalid plain text segment: {line}')
                continue

            speaker, text = m.groups()
            speaker = speaker.strip()
            text = text.strip()
            if not speaker or not text:
                LOG.warning(f'Invalid plain text segment: {line}')
                continue

            segments.append(SegmentSource(id=f'{len(segments)}', speaker_id=speaker, text=text))
            speakers.add(speaker)

        return Transcript(
            id=self._transcript_id,
            language=self._language or 'und',
            segments=segments,
            url=self._url,
        )


class ExportHelper:
    def __init__(self, article: Article) -> None:
        self.article = article
        # [type group, list of paragraphs], e.g. 'title' -> [title_para_1, title_para_2]
        # different groups: 'title', 'summary', 'lead', 'q_and_a', 'conclusion'
        self.paragraphs: dict[str, list[Paragraph]] = defaultdict(list)

    def _load_paragraphs(self):
        for paragraph in self.article.paragraphs:
            if paragraph.type in ('question', 'answer', 'text'):
                self.paragraphs['q_and_a'].append(paragraph)
            else:
                self.paragraphs[paragraph.type].append(paragraph)

    def get_titles(self) -> tuple[str, list[str]]:
        """Returns tuple with main title text and list of alternative titles texts"""
        if not self.paragraphs:
            self._load_paragraphs()
        title_paragraphs: list[str] = [p.text for p in self.paragraphs['title']]
        if not title_paragraphs:
            LOG.warning('No title paragraphs were found using Article ID!')
            return self.article.id, []
        else:
            return title_paragraphs[0], title_paragraphs[1:]

    def get_speakers(self) -> Iterable[str]:
        """Returns iterator of speakers (present in the Q&A part) also with their role and description assigned"""
        if not self.paragraphs:
            self._load_paragraphs()
        q_and_a_speakers: set[str] = {paragraph.speaker for paragraph in self.paragraphs['q_and_a'] if
                                      paragraph.speaker}

        transcript_speakers: list[Speaker] = []
        if self.article.transcript:
            reference_transcript: Transcript = Transcript.model_validate(self.article.transcript)
            transcript_speakers = reference_transcript.speakers
        if transcript_speakers:
            for speaker in transcript_speakers:
                # Skip speakers not present in the Q&A part
                if speaker.speaker_id not in q_and_a_speakers:
                    continue
                name = speaker.name or speaker.speaker_id
                role = speaker.role
                if speaker.description:
                    role = f'{speaker.description}'
                yield f'{name} ({role})'
        else:
            for speaker in q_and_a_speakers:
                yield speaker

    def get_qa_paragraphs(self) -> Iterable[tuple[str, str | None, str | None]]:
        """Returns iterator of tuples with paragraph text and optionally speaker name and topic ID"""
        if self.article.transcript:
            transcript = Transcript.model_validate(self.article.transcript)
            speakers_by_id = {speaker.speaker_id: speaker for speaker in transcript.speakers}
        else:
            speakers_by_id = None

        if not self.paragraphs:
            self._load_paragraphs()
        # topic_id -> list of corresponding paragraphs
        para_by_topic: dict[str, list[Paragraph]] = defaultdict(list)
        topics_by_id: dict[str, Topic] = {topic.id: topic for topic in self.article.topics.topics}
        topics_in_order: list[str] = []
        for paragraph in self.paragraphs['q_and_a']:
            if paragraph.topic_id not in topics_in_order:
                topics_in_order.append(paragraph.topic_id)
            para_by_topic[paragraph.topic_id].append(paragraph)

        for topic in topics_in_order:
            for paragraph in para_by_topic[topic]:
                if paragraph.type == 'text':
                    speaker_name = None
                elif speakers_by_id and (speaker := speakers_by_id.get(paragraph.speaker)):
                    speaker_name = speaker.name or speaker.speaker_id
                else:
                    speaker_name = paragraph.speaker
                text = paragraph.text
                if paragraph.type == 'answer' and not (text.startswith('"') and text.endswith('"')):
                    text = f'"{paragraph.text}"'
                # Add source timecodes
                if timecodes := paragraph.source_timecodes:
                    text = f'{text} [{round(timecodes.start_time, 2)}, {round(timecodes.end_time, 2)}]'
                else:
                    LOG.warning(f'Paragraph {paragraph.id} is missing timecodes')
                yield text, speaker_name, topics_by_id[topic].title

    def get_summary_points(self) -> Iterable[str]:
        """Extracts summary bullet points from the first 'summary' paragraph, if any.

        The summary paragraph text is expected to contain multiple lines, typically starting with
        '- ' for bullets, but we accept any non-empty trimmed lines.
        """

        if not self.paragraphs:
            self._load_paragraphs()
        if self.paragraphs['summary'] and len(summary := self.paragraphs['summary']) >= 1:
            for raw_line in summary[0].text.splitlines():
                if not (line := raw_line.strip()):
                    continue
                # remove common bullet prefixes
                if line[:2] in ('- ', '• ', '* '):
                    line = line[2:].strip()
                yield line
        else:
            LOG.warning('Zero or more than one summary paragraphs were found, exactly one is required, returning '
                        'empty summary.')

    def get_lead_paras(self) -> Iterable[str]:
        if not self.paragraphs:
            self._load_paragraphs()
        for paragraph in self.paragraphs['lead']:
            yield paragraph.text
            yield ''

    def get_conclusion_texts(self) -> Iterable[str]:
        if not self.paragraphs:
            self._load_paragraphs()
        for paragraph in self.paragraphs['conclusion']:
            yield paragraph.text


class MarkdownExporter:
    def __init__(self):
        self.notes: list[str] = []

    @staticmethod
    def _get_titles(export_helper: ExportHelper) -> Iterable[str]:
        main_title, alternative_titles = export_helper.get_titles()
        yield f'# {main_title}'

        if alternative_titles:
            yield '## Alternative Titles'
            for alt_title in alternative_titles:
                yield f'- **{alt_title}**'
        yield ''

    @staticmethod
    def _get_speakers(export_helper: ExportHelper) -> Iterable[str]:
        yield '---'
        yield '### Speakers'
        yield ''
        for speaker in export_helper.get_speakers():
            yield f'- {speaker}'
        yield '---'

    @staticmethod
    def _get_summary(export_helper: ExportHelper) -> Iterable[str]:
        yield '### Summary'
        yield ''
        for point in export_helper.get_summary_points():
            yield f'- {point}'
        yield '---'

    @staticmethod
    def _get_lead(export_helper: ExportHelper) -> Iterable[str]:
        for para in export_helper.get_lead_paras():
            yield para
            yield ''

    @staticmethod
    def _get_qa_paragraphs(export_helper: ExportHelper) -> Iterable[str]:
        last_topic_title = None
        for text, speaker, topic_title in export_helper.get_qa_paragraphs():
            if last_topic_title != topic_title:
                if topic_title:
                    yield ''
                    yield f'### {topic_title}'
                    yield ''
                last_topic_title = topic_title
            if speaker:
                yield f'**{speaker}:** {text}'
            else:
                yield text
            yield ''

    @staticmethod
    def _get_conclusion(export_helper) -> Iterable[str]:
        for para in export_helper.get_conclusion_texts():
            yield para
            yield ''

    def _add_web_extracted_metadata(self, article: Article, web_searches: list[WebSearchResult]) -> None:
        article_paras_by_id: dict[str, Paragraph] = {p.id: p for p in article.paragraphs}
        for note_idx, additional_info in enumerate(web_searches):
            sources_list = additional_info.sources or []
            sources_block = ''
            if sources_list:
                sources = '\n'.join(f'-  {source.url}' for source in sources_list)
                sources_block = f'\n<br>**Sources**:\n{sources}'
            note_text = additional_info.summary or ''
            ai_note = f'{note_text}{sources_block}'
            self.notes.append(ai_note)
            for text_occurrence in additional_info.text_occurrences:
                if not (paragraph := article_paras_by_id.get(text_occurrence.paragraph_id)):
                    LOG.warning(f'Text occurrence {text_occurrence.text_string} refences invalid paragraph ID '
                                f'{text_occurrence.paragraph_id}')
                    continue
                else:
                    escaped_text = re.escape(text_occurrence.text_string)
                    unlinked_pattern = re.compile(rf'{escaped_text}(?!]\(#ai-note-\d+\))')
                    if not unlinked_pattern.search(paragraph.text):
                        if text_occurrence.text_string not in paragraph.text:
                            LOG.warning(f'Text occurrence {text_occurrence.text_string} was not found in paragraph '
                                        f'{text_occurrence.paragraph_id}')
                    else:
                        new_text = f'[{text_occurrence.text_string}](#ai-note-{note_idx + 1})'
                        paragraph.text = unlinked_pattern.sub(new_text, paragraph.text, count=1)

    def include_metadata(self, article: Article) -> Article:
        """Enriches the article text using additional info/context from the metadata"""
        if metadata := article.metadata:
            if metadata.web_searches and metadata.web_searches.data:
                self._add_web_extracted_metadata(article, metadata.web_searches.data)
        else:
            LOG.warning(f'Metadata are empty for article {article.id}')
        return article

    def _get_notes(self) -> Iterable[str]:
        yield '---'
        yield '### Notes'
        yield ''
        for idx, note in enumerate(self.notes):
            yield f'#### AI Note {idx + 1}'
            yield note

    def to_lines(self, article: Article) -> Iterable[str]:
        self.notes = []
        article = self.include_metadata(article)
        export_helper = ExportHelper(article)
        for title in self._get_titles(export_helper):
            yield title

        for speaker_part in self._get_speakers(export_helper):
            yield speaker_part

        # Summary section placed after Speakers and before content
        for summary_part in self._get_summary(export_helper):
            yield summary_part

        for lead_part in self._get_lead(export_helper):
            yield lead_part

        for qa_part in self._get_qa_paragraphs(export_helper):
            yield qa_part

        for line in self._get_conclusion(export_helper):
            yield line

        for note in self._get_notes():
            yield note


class GoogleDocExporter:
    def __init__(self, google_folder: str | None = None) -> None:
        self.google_folder: str | None = google_folder
        self.speakers: set[str] = set()
        self.titles: tuple[str, list[str] | None] | None = None
        self.topic_headings: set[str] = set()
        self.note_headings: list[str] = list()
        # tuples [web page title, web page URL]
        self.sources: list[tuple[str, str]] = list()

    def _get_titles(self, export_helper: ExportHelper) -> list[str]:
        main_title, alternative_titles = export_helper.get_titles()
        self.titles = main_title, alternative_titles
        lines: list[str] = [main_title, 'Alternative Titles']
        if alternative_titles:
            for alternative in alternative_titles:
                lines.append(f'   -   {alternative}')
        lines.append('')
        return lines

    @staticmethod
    def _get_speakers(export_helper: ExportHelper) -> list[str]:
        lines: list[str] = ['-------------------------------------------------------------------------------------',
                            'Speakers']
        speakers = export_helper.get_speakers()
        for speaker in speakers:
            lines.append(f'   -   {speaker}')
        lines.append('')
        lines.append('-------------------------------------------------------------------------------------')
        lines.append('')
        return lines

    @staticmethod
    def _get_summary(export_helper: ExportHelper) -> list[str]:
        lines: list[str] = ['Summary']
        points = export_helper.get_summary_points()
        for p in points:
            lines.append(f'   -   {p}')
        lines.append('')
        lines.append('-------------------------------------------------------------------------------------')
        lines.append('')
        return lines

    def _get_qa_paragraphs(self, export_helper: ExportHelper) -> list[str]:
        lines: list[str] = list(export_helper.get_lead_paras())
        last_topic_title: str | None = None
        for text, speaker, topic_title in export_helper.get_qa_paragraphs():
            if last_topic_title != topic_title:
                lines.append(topic_title)
                self.topic_headings.add(topic_title)
                last_topic_title = topic_title
            if speaker:
                lines.append(f'{speaker}: {text}')
                self.speakers.add(f'{speaker}:')
            else:
                lines.append(text)
            lines.append('')
        return lines

    @staticmethod
    def _get_conclusion(export_helper: ExportHelper) -> list[str]:
        lines: list[str] = []
        texts = export_helper.get_conclusion_texts()
        for t in texts:
            lines.append(t)
            lines.append('')
        return lines

    def _get_notes(self, article: Article) -> list[str]:
        if not article.metadata or not article.metadata.web_searches or not article.metadata.web_searches.data:
            return []
        lines: list[str] = ['-------------------------------------------------------------------------------------',
                            'Notes', '']
        for idx, note in enumerate(article.metadata.web_searches.data):
            heading = f'AI Note {idx + 1}'
            self.note_headings.append(heading)
            lines.append(heading)
            sources_list = note.sources
            self.sources += [(f'- {source.title}', source.url) for source in sources_list]
            sources = '\n'.join(f'  - {source.title}' for source in sources_list)
            note_text = note.summary
            ai_note = f'{note_text}\nSources:\n{sources}'
            lines.append(ai_note)
            lines.append('')
        return lines

    @staticmethod
    def get_google_doc_paras_offset(request_wrapper: GoogleDocRequestsWrapper) -> dict[str, tuple[int, int]]:
        """Returns dict with paragraph texts as keys and tuples with character [startIndex, endIndex] as values."""
        google_doc_paras: list[dict[str, Any]] = request_wrapper.get_document_paragraphs()
        paragraph_offsets: dict[str, tuple[int, int]] = dict()
        for paragraph in google_doc_paras:
            start_index = paragraph['startIndex']
            end_index = paragraph['endIndex']
            for element in dictutil.getValue(paragraph, 'paragraph.elements', []):
                if content := dictutil.getValue(element, 'textRun.content'):
                    paragraph_offsets[content.strip()] = (start_index, end_index)
        return paragraph_offsets

    @staticmethod
    def _get_summary_paragraph_end_index(paragraph_by_id: dict[str, Paragraph],
                                         google_doc_paras_offset: dict[str, tuple[int, int]]) -> int:
        """Returns end index of summary paragraph as start index of the lead paragraph (which always follows) - 1."""
        lead_paragraphs = [p for p in paragraph_by_id.values() if p.type == 'lead']
        if not lead_paragraphs:
            raise ValueError('No lead paragraphs found.')
        first_lead = min(lead_paragraphs, key=lambda p: p.id)

        if not (indexes := google_doc_paras_offset.get(first_lead.text)):
            raise ValueError(f'Lead paragraph {first_lead.id} text not found in Google doc paragraphs.')
        return indexes[0] - 1

    def add_links_to_text_occurrence(self,
                                     text: TextOccurrence,
                                     heading_id: str,
                                     paragraph_by_id: dict[str, Paragraph],
                                     google_doc_paras_offset: dict[str, tuple[int, int]],
                                     request_wrapper: GoogleDocRequestsWrapper) -> None:
        if not (paragraph := paragraph_by_id.get(text.paragraph_id)):
            LOG.warning(f'Text occurrence {text.text_string} refences invalid paragraph ID {text.paragraph_id}')
            return
        # Because the text in Google Doc paragraphs is a bit modified (new lines, extra quotes,
        # speaker name/id appended, ...), in comparison to original paragraphs (but some substring should always match)
        # we loop through them
        for key in google_doc_paras_offset.keys():
            # Summary paragraph contains new lines, which breaks the Google Doc paragraph into multiple ones, it is
            # easier to search here only for the string 'Summary' and get the endIndex from the lead paragraph, which
            # always follows the summary, rather than splitting the original summary paragraph and searching to which
            # bullet the given text occurrence belongs to
            paragraph_text = 'Summary' if paragraph.type == 'summary' else paragraph.text
            if paragraph_text in key:
                start_index, end_index = google_doc_paras_offset[key]
                end_index = self._get_summary_paragraph_end_index(
                    paragraph_by_id, google_doc_paras_offset) if paragraph.type == 'summary' else end_index
                request_wrapper.update_text_style_request(
                    texts=[text.text_string],
                    heading_link=heading_id,
                    search_index_start=start_index,
                    search_index_end=end_index,
                    find_all=True
                )
                return
        LOG.warning(f'Paragraph {paragraph.id} with text:\n{paragraph.text}\n, was not found among the '
                    f'paragraphs in Google doc.')

    def _link_note_occurrences(self, request_wrapper: GoogleDocRequestsWrapper, article: Article) -> None:
        if not article.metadata or not article.metadata.web_searches or not article.metadata.web_searches.data:
            return
        heading_ids = request_wrapper.get_heading_ids_by_titles(self.note_headings)
        google_doc_paras_offset = self.get_google_doc_paras_offset(request_wrapper)
        paragraph_by_id: dict[str, Paragraph] = {p.id: p for p in article.paragraphs}
        for web_info_id, web_info in enumerate(article.metadata.web_searches.data):
            heading_id = heading_ids[web_info_id]
            if not heading_id:
                LOG.warning(f'Heading ID missing for AI Note {web_info_id + 1}, link was not added.')
                continue
            for text in web_info.text_occurrences:
                self.add_links_to_text_occurrence(text, heading_id, paragraph_by_id,
                                                  google_doc_paras_offset, request_wrapper)

    def _link_source_urls(self, request_wrapper: GoogleDocRequestsWrapper) -> None:
        for source in self.sources:
            request_wrapper.update_text_style_request(
                texts=[source[0]],
                url_link=source[1],
                find_all=True,
                skip_formatted_paragraphs=True,
            )

    @staticmethod
    def link_segment_timecodes(request_wrapper: GoogleDocRequestsWrapper, article: Article) -> None:
        if not article.transcript or not (url := article.transcript.url):
            return
        host = (url.host or '').lower()
        path = url.path or ''
        has_query = bool(url.query)
        is_youtube_short = host.endswith('youtu.be')
        is_youtube_watch = host.endswith('youtube.com') and path == '/watch'
        if not (is_youtube_short or is_youtube_watch):
            LOG.warning(f'Transcript URL {url} does not point to YouTube video, timecodes linking skipped.')
            return
        url_string = str(url)
        if url.fragment:
            url_base, fragment = url_string.split('#', 1)
            fragment_suffix = f'#{fragment}'
        else:
            url_base = url_string
            fragment_suffix = ''
        for paragraph in article.paragraphs:
            if paragraph.type not in ('question', 'answer', 'text'):
                continue
            if not paragraph.source_timecodes:
                LOG.warning(f'Paragraph {paragraph.id} has no source timecodes, skipping.')
                continue

            start_seconds = int(paragraph.source_timecodes.start_time)
            separator = '&' if has_query else '?'
            if is_youtube_short:
                timecode_url = f'{url_base}{separator}t={start_seconds}{fragment_suffix}'
            elif is_youtube_watch:
                timecode_url = f'{url_base}{separator}t={start_seconds}s{fragment_suffix}'
            else:
                continue    # Should never reach this path
            request_wrapper.update_text_style_request(
                texts=[f'[{round(paragraph.source_timecodes.start_time, 2)}, '
                       f'{round(paragraph.source_timecodes.end_time, 2)}]'],
                url_link=timecode_url,
                find_all=True,
                skip_formatted_paragraphs=True,
            )

    def article_to_text(self, article: Article) -> str:
        export_helper = ExportHelper(article)
        lines = self._get_titles(export_helper)
        lines += self._get_speakers(export_helper)
        # Summary section after speakers
        lines += self._get_summary(export_helper)
        lines_qa_section = self._get_qa_paragraphs(export_helper)
        lines += lines_qa_section
        # Conclusion at the end
        lines += self._get_conclusion(export_helper)
        lines += self._get_notes(article)

        while lines and lines[-1] == '':
            lines.pop()
        return '\n'.join(lines)

    def _set_formatting(self, article: Article, request_wrapper: GoogleDocRequestsWrapper) -> None:
        request_wrapper.update_all_paragraphs_style_request(line_spacing=1.15)
        # Set main title to Heading 1 style
        if not self.titles:
            LOG.warning(f'No titles found for article {article.id}')
        else:
            request_wrapper.update_paragraph_style_request(texts=[self.titles[0]], named_style='HEADING_1')
            # Set alternative titles to bold format
            if alternative_titles := self.titles[1]:
                request_wrapper.update_text_style_request(texts=alternative_titles, text_style='bold')
        # Other headings formatting
        request_wrapper.update_paragraph_style_request(texts=['Alternative Titles'], named_style='HEADING_2')
        request_wrapper.update_paragraph_style_request(texts=['Speakers'], named_style='HEADING_3')
        request_wrapper.update_paragraph_style_request(texts=['Summary'], named_style='HEADING_3')
        if self.topic_headings:
            request_wrapper.update_paragraph_style_request(texts=list(self.topic_headings), named_style='HEADING_3')
        if self.note_headings:
            request_wrapper.update_paragraph_style_request(texts=['Notes'], named_style='HEADING_3')
            request_wrapper.update_paragraph_style_request(texts=self.note_headings, named_style='HEADING_4')
        # Committing current updates, allows to update paragraph styles and skip headings, etc. when making the speaker
        # names bold in the Q&A part
        request_wrapper.commit_requests()
        # Set all speakers in the Q&A section to bold format
        if not self.speakers:
            LOG.warning(f'No speakers found for article {article.id}')
        else:
            request_wrapper.update_text_style_request(texts=list(self.speakers), text_style='bold', find_all=True,
                                                      skip_formatted_paragraphs=True)

    def generate_article(self, article: Article, request_wrapper: GoogleDocRequestsWrapper) -> None:
        # Reset the instance attributes
        self.speakers = set()
        self.titles = None
        self.topic_headings = set()
        self.note_headings = list()
        self.sources = list()
        # Create doc
        request_wrapper.create_document_request(article.id, parent_folder_id=self.google_folder)
        # Get the article text and insert it into Google doc
        text = self.article_to_text(article)
        request_wrapper.insert_text_request(text)
        # Adjust the formatting
        self._set_formatting(article, request_wrapper)
        self._link_note_occurrences(request_wrapper, article)
        self._link_source_urls(request_wrapper)
        self.link_segment_timecodes(request_wrapper, article)


def import_cli(args):
    if args.format == 'plain-text':
        with cliutil.argOpenOut(args) as fout:
            if args.input_file is not None:
                # assert isinstance(args.input_file, Path), f'Expected Path, but got: {type(args.input_file)}'
                input_path = Path(args.input_file)
                if glob.escape(args.input_file) != args.input_file:
                    # glob pattern
                    if input_path.is_absolute():
                        search_root = input_path.anchor
                    else:
                        search_root = Path('.')

                    for p in search_root.glob(args.input_file):
                        with p.open(mode='r', encoding='utf-8') as finp:
                            importer = PlainTextImporter(transcript_id=p.name, language=args.language)
                            transcript = importer.from_lines(finp)
                            print(transcript.model_dump_json(), file=fout)

                else:
                    # single file
                    with cliutil.argOpenIn(args) as finp:
                        importer = PlainTextImporter(transcript_id=input_path.name, language=args.language)
                        transcript = importer.from_lines(finp)
                        print(transcript.model_dump_json(), file=fout)
            else:
                # stdio
                with cliutil.argOpenIn(args) as finp:
                    importer = PlainTextImporter(transcript_id=uuid.uuid4().hex, language=args.language)
                    transcript = importer.from_lines(finp)
                    print(transcript.model_dump_json(), file=fout)

    else:
        raise ValueError(f'Unsupported format: {args.format}')


def export_markdown(args):
    exporter = MarkdownExporter()
    with cliutil.argOpenIn(args) as finp:
        if args.output_file is not None:
            output_path = Path(args.output_file)
            if output_path.is_dir():
                # each article in a separate file
                for article in map(Article.model_validate_json, finp):
                    path = output_path / f'{article.id}.md'
                    with path.open(mode='w', encoding='utf-8') as fout:
                        for line in exporter.to_lines(article):
                            print(line, file=fout)

            else:
                # all articles to a single file
                with cliutil.argOpenOut(args) as fout:
                    for article in map(Article.model_validate_json, finp):
                        for line in exporter.to_lines(article):
                            print(line, file=fout)

                        print('', file=fout)
                        print('---', file=fout)
                        print('', file=fout)

        else:
            # stdout
            with cliutil.argOpenOut(args) as fout:
                for article in map(Article.model_validate_json, finp):
                    for line in exporter.to_lines(article):
                        print(line, file=fout)

                    print('', file=fout)
                    print('---', file=fout)
                    print('', file=fout)


def export_google_doc(args):
    if args.token:
        request_wrapper = GoogleDocRequestsWrapper(client_token=args.token, batch_requests=True)
    elif args.credentials:
        request_wrapper = GoogleDocRequestsWrapper(service_credentials_path=args.credentials, batch_requests=True)
    else:
        raise ValueError('Must provide either --token or --credentials to access Google Docs API!')
    with cliutil.argOpenIn(args) as finp:
        google_exporter = GoogleDocExporter(args.google_folder)
        for article in map(Article.model_validate_json, finp):
            google_exporter.generate_article(article, request_wrapper)
            request_wrapper.commit_requests()


def export_cli(args):
    if args.format == 'markdown':
        export_markdown(args)
    elif args.format == 'google-doc':
        export_google_doc(args)
    else:
        raise ValueError(f'Unsupported format: {args.format}')


def main():
    parser = argparse.ArgumentParser(
        description='Converts various formats to Transcript JSON format and from Article JSON format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='available commands')

    p_import = subparsers.add_parser('import', formatter_class=parser.formatter_class,
                                     help='Converts to the Transcript JSON format.')
    p_import.add_argument('--format', choices=['plain-text'], default='plain-text',
                          help='The format of the input file.')
    p_import.add_argument('--language', help='The transcription text language.')
    cliutil.addInOutArgGroups(p_import)
    p_import.set_defaults(method=import_cli)

    p_export = subparsers.add_parser('export', formatter_class=parser.formatter_class,
                                     help='Converts from the Article JSON format.')
    p_export.add_argument('--format', choices=['markdown', 'google-doc'], default='markdown',
                          help='The format of the output file. If the google-doc is selected path to an access token '
                               'or credentials JSONl file must be provided.')
    p_export.add_argument('--token', type=Path,
                          help='Path to an access token for Google Docs API.'
                               'See https://developers.google.com/workspace/docs/api/quickstart/python')
    p_export.add_argument('--credentials', type=Path,
                          help='Path to a Google service account JSON file.'
                               'See https://developers.google.com/workspace/docs/api/quickstart/python')
    p_export.add_argument('--google-folder', type=str, default=None,
                          help='ID of a folder on Google Drive for storing the generated articles to.')
    cliutil.addInOutArgGroups(p_export)
    p_export.set_defaults(method=export_cli)

    logutil.addLogArguments(parser)
    args = parser.parse_args()
    logutil.configureFromArgs(args)

    if args.method:
        args.method(args)
    else:
        parser.error('Unknown command')


if __name__ == '__main__':
    main()
