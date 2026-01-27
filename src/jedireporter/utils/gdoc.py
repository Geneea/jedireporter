# coding=utf-8
"""
Provides access to Google Docs. It allows creating a document, adding text to it and add specific formatting.
It creates wrapper around main function ``batchUpdate`` from Google Docs API, which requires direct JSON payload.

See documentation here:
 * Google Docs API -- https://developers.google.com/workspace/docs/api/reference/rest
 * Python Quickstart (also how to get credentials) -- https://developers.google.com/workspace/docs/api/quickstart/python
"""

import re
import time
from pathlib import Path
from typing import Any

import googleapiclient.discovery
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.errors import HttpError

from jedireporter.utils import dict_utils as dictutil
from jedireporter.utils import logging as logutil

LOG = logutil.getLogger(__package__, __file__)


class GoogleDocRequestsWrapper:
    SCOPES = ['https://www.googleapis.com/auth/documents',
              'https://www.googleapis.com/auth/drive.file']

    def __init__(
            self,
            *,
            batch_requests: bool = False,
            credentials: Credentials | None = None,
            service_credentials_path: Path | None = None,
            client_token: Path | None = None
    ) -> None:
        self.document_id: str | None = None
        if credentials:
            self._credentials = credentials
        elif service_credentials_path:
            self._credentials = ServiceAccountCredentials.from_service_account_file(str(service_credentials_path))
        elif client_token and (credentials := self._get_credentials_from_token(client_token)):
            self._credentials = credentials
        else:
            raise ValueError('No valid authentication method was provided.')
        # Disable discovery cache to avoid oauth2client file_cache warnings
        self._service = googleapiclient.discovery.build(
            'docs', 'v1', credentials=self._credentials, cache_discovery=False)
        self._drive_service = googleapiclient.discovery.build(
            'drive', 'v3', credentials=self._credentials, cache_discovery=False)
        self._batch_requests: bool = batch_requests
        self._awaiting_requests: dict[str, list[Any]] = {'requests': []}

    # --- Authentication & credentials ---
    def _get_credentials_from_token(self, access_token_path: Path) -> Credentials | None:
        if access_token_path.exists():
            credentials = Credentials.from_authorized_user_file(str(access_token_path), self.SCOPES)
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            return credentials
        return None

    # --- Request execution with retry handling ---
    def commit_requests(self):
        """If the class instance is initialized with `batch_requests` argument, non-inserting requests are stored and
        awaits call of this method to be sent to Google API."""
        self._execute_batch_update(body=self._awaiting_requests)
        self._awaiting_requests: dict[str, list[Any]] = {'requests': []}

    def _execute_batch_update(self, body: dict[str, Any]) -> None:
        """Run a batchUpdate with retry/backoff when we hit write rate limits."""
        max_retries = 5
        base_sleep = 5.0
        for attempt in range(max_retries):
            try:
                self._service.documents().batchUpdate(documentId=self.document_id, body=body).execute()
                return
            except HttpError as exc:
                status = getattr(getattr(exc, 'resp', None), 'status', None)
                if status != 429 or attempt == max_retries - 1:
                    raise
                sleep_for = min(60.0, base_sleep * (2 ** attempt))
                LOG.warning(f'Google Docs write rate limit hit (429). Retrying in {sleep_for:.1f}s...')
                time.sleep(sleep_for)

    # --- Helper methods ---
    @staticmethod
    def _find_text_in_paragraph(paragraph: dict[str, Any], text: str,
                                search_index_start: int | None = None,
                                search_index_end: int | None = None,
                                skip_formatted_paragraphs: bool = False) -> list[int]:
        # Skip already adjusted paragraphs, i.e., headings etc.
        if skip_formatted_paragraphs and dictutil.getValue(
                paragraph, 'paragraphStyle.namedStyleType', 'NORMAL_TEXT') != 'NORMAL_TEXT':
            return []
        start_indexes: list[int] = []
        for element in paragraph.get('elements', []):
            # Search the text from a given `search_index_start` position
            if search_index_start is not None and element['endIndex'] <= search_index_start:
                continue
            # Finish the search once the `search_index_end` position is reached
            if search_index_end is not None and element['startIndex'] >= search_index_end:
                return start_indexes
            pattern = re.escape(text)
            for match in re.finditer(pattern, dictutil.getValue(element, 'textRun.content', '')):
                doc_index = match.start() + element.get('startIndex', 0)
                if search_index_start is not None and doc_index < search_index_start:
                    continue
                start_indexes.append(doc_index)
        return start_indexes

    def _find_text_in_table_cell(self, cell: dict[str, Any], text: str,
                                 search_index_start: int | None = None,
                                 search_index_end: int | None = None,
                                 skip_formatted_paragraphs: bool = False) -> list[int]:
        start_indexes: list[int] = []
        for cell_content in cell.get('content', []):
            if 'paragraph' in cell_content:
                start_indexes += self._find_text_in_paragraph(
                    cell_content['paragraph'], text, search_index_start, search_index_end, skip_formatted_paragraphs)
        return start_indexes

    def _find_text_in_table(self, table: dict[str, Any], text: str, search_index_start: int | None = None,
                            search_index_end: int | None = None,
                            skip_formatted_paragraphs: bool = False) -> list[int]:
        start_indexes: list[int] = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                found = self._find_text_in_table_cell(cell, text, search_index_start, search_index_end,
                                                      skip_formatted_paragraphs)
                start_indexes += found
        return start_indexes

    def _find_text_in_doc(self, text: str, *, search_index_start: int | None = None,
                          search_index_end: int | None = None, find_all: bool = False,
                          skip_formatted_paragraphs: bool = False) -> list[int]:
        """
        Searches for occurrences of text in the body['content'] of the Google Doc, only elements of type 'paragraph'
        and 'table' are inspected.

        :param text: given string, that is searched in the document
        :param search_index_start: if set, the search starts from a given index, all occurrences before are ignored
        :param search_index_end: if set, the search ends at a given index, all occurrences after are ignored
        :param find_all: if set, all non-overlapping matches are returned, otherwise only the first match
        :param skip_formatted_paragraphs: if set, any formatted paragraphs will be skipped (i.e. Headings, etc.)
        :return: list with start indexes of found occurrences of the given text string
        """
        start_indexes: list[int] = []
        content = self._get_body_content()
        for doc_elem in content:
            if search_index_start is not None and doc_elem['endIndex'] < search_index_start:
                continue
            if 'paragraph' in doc_elem:
                start_indexes += self._find_text_in_paragraph(
                    doc_elem['paragraph'], text, search_index_start, search_index_end, skip_formatted_paragraphs)
            elif 'table' in doc_elem:
                start_indexes += self._find_text_in_table(
                    doc_elem['table'], text, search_index_start, search_index_end, skip_formatted_paragraphs)
            if not find_all and start_indexes:
                break
        return start_indexes

    def _get_body_content(self) -> list[dict[str, Any]]:
        doc_state = self._service.documents().get(documentId=self.document_id).execute()
        return [element for element in dictutil.getValue(doc_state, 'body.content', [])]

    def _get_ith_table_of_document(self, table_idx: int) -> dict[str, Any]:
        body_content = self._get_body_content()
        found_tables = 0
        for element in body_content:
            if 'table' in element:
                if found_tables == table_idx:
                    return element
                else:
                    found_tables += 1
        raise IndexError(f'Table idx {table_idx} is out of range!')

    @staticmethod
    def _list_paragraphs_in_table(table: dict[str, Any]) -> list[dict[str, Any]]:
        paragraphs = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                for content in cell.get('content', []):
                    if 'paragraph' in content:
                        paragraphs.append(content)
        return paragraphs

    # --- Miscellaneous public helpers ---
    def get_cell_index_from_table(self, table_idx: int, row: int, col: int) -> int:
        """Based on the table index (0-based index of tables in the document, starting from the top of the document)
        and row and col indexes returns cell index (character-level offset in the document)"""
        table = self._get_ith_table_of_document(table_idx)
        if (col >= dictutil.getValue(table, 'table.columns', default=0) or
                row >= dictutil.getValue(table, 'table.rows', default=0)):
            raise IndexError(f'Selected indexes [{row}, {col}] are out of range for table with '
                             f'{dictutil.getValue(table, "table.rows")} rows and '
                             f'{dictutil.getValue(table, "table.columns")} columns!')
        cell_start_index = dictutil.getValue(table, f'table.tableRows.[{row}].tableCells.[{col}].startIndex')
        if cell_start_index is None:
            raise KeyError(f'startIndex not found for table[{row}][{col}]')
        return cell_start_index + 1

    def get_ith_table_doc_range(self, table_idx: int) -> tuple[int, int]:
        """Based on the table index (0-based index of tables in the document, starting from the top of the document)
        returns start and end index of table (character-level offset in the document)"""
        table = self._get_ith_table_of_document(table_idx)
        return table['startIndex'], table['endIndex']

    def get_heading_ids_by_titles(self, heading_titles: list[str], *, require_heading: bool = True,
                                  search_index_start: int | None = None,
                                  search_index_end: int | None = None) -> list[str]:
        content = self._get_body_content()
        heading_title_to_id: dict[str, str] = {}

        def _process_paragraph(paragraph: dict[str, Any]) -> None:
            named_style = dictutil.getValue(paragraph, 'paragraphStyle.namedStyleType', 'NORMAL_TEXT')
            if require_heading and named_style == 'NORMAL_TEXT':
                return
            para_end_index = dictutil.getValue(paragraph, 'elements.[0].endIndex')
            if search_index_start is not None and para_end_index and para_end_index <= search_index_start:
                return
            para_start_index = dictutil.getValue(paragraph, 'elements.[0].startIndex')
            if para_start_index and search_index_end and para_start_index >= search_index_end:
                return
            elif text := dictutil.getValue(paragraph, 'elements.[0].textRun.content'):
                if text.strip() in heading_title_to_id:
                    return    # prefer the first found heading
                heading_title_to_id[text.strip()] = dictutil.getValue(paragraph, 'paragraphStyle.headingId')

        for doc_elem in content:
            if doc_elem.get('startIndex') and search_index_end and doc_elem.get('startIndex') >= search_index_end:
                break
            if table := dictutil.getValue(doc_elem, 'table'):
                for paragraph in self._list_paragraphs_in_table(table):
                    _process_paragraph(paragraph['paragraph'])
            elif paragraph := doc_elem.get('paragraph'):
                _process_paragraph(paragraph)
            else:
                continue

        heading_ids = [heading_title_to_id.get(heading_title) for heading_title in heading_titles]
        if not all(heading_ids):
            none_element = heading_ids.index(None)
            raise ValueError(f'Heading title {heading_titles[none_element]} not found in the document!')
        else:
            return heading_ids

    def get_document_paragraphs(self):
        paragraphs: list[dict] = []
        content = self._get_body_content()
        for doc_elem in content:
            if 'paragraph' in doc_elem:
                paragraphs.append(doc_elem)
            elif 'table' in doc_elem:
                paragraphs += self._list_paragraphs_in_table(doc_elem['table'])
        return paragraphs

    # --- Requests with adding elements to doc ---
    def create_document_request(self, document_name: str, flip_orientation: bool = False,
                                parent_folder_id: str | None = None) -> str:
        if parent_folder_id:
            # Drive API supports placing the doc directly into a folder via `parents`.
            file = self._drive_service.files().create(
                body={
                    'name': document_name,
                    'mimeType': 'application/vnd.google-apps.document',
                    'parents': [parent_folder_id],
                },
                fields='id',
            ).execute()
            self.document_id = file.get('id')
        else:
            document = self._service.documents().create(body={'title': document_name}).execute()
            self.document_id = document.get('documentId')
        if flip_orientation:
            self.update_document_style_request(document_style='flipPageOrientation')
        return self.document_id

    def insert_text_request(self, text: str, index: int | None = None) -> None:
        """If index is provided, insert text into the specified place, otherwise, inserts at the end of the document."""
        if index is None:
            if content := self._get_body_content():
                index = content[-1]['endIndex'] - 1
            else:
                index = 1
        request = {'insertText': {'location': {'index': index}, 'text': text}}
        self._execute_batch_update({'requests': [request]})

    def insert_table_request(self, rows_nbr: int, cols_nbr: int) -> None:
        request = {'insertTable': {
            'rows': rows_nbr,
            'columns': cols_nbr,
            'endOfSegmentLocation': {
                'segmentId': ''}}}
        self._execute_batch_update({'requests': [request]})

    def fill_table_cell_value_request(self, table_idx: int, row: int, col: int, text: str) -> None:
        """Based on the table index (0-based index of tables in the document, starting from the top of the document)
        and row and col indexes fills the specified cell with the given text"""
        cell_start_index = self.get_cell_index_from_table(table_idx, row, col)
        self.insert_text_request(text=text, index=cell_start_index)

    # --- Requests updating formatting ---
    def update_document_style_request(self, document_style: str) -> None:
        request: dict[str, Any] = {'updateDocumentStyle': {
            'documentStyle': {
                f'{document_style}': True
            },
            'fields': f'{document_style}'
        }}
        if self._batch_requests:
            self._awaiting_requests['requests'].append(request)
        else:
            self._execute_batch_update({'requests': [request]})

    def update_paragraph_style_request(self, texts: list[str], *, named_style: str | None = None,
                                       line_spacing: float | None = None, search_index_start: int | None = None,
                                       skip_formatted_paragraphs: bool = False) -> None:
        """
        Searches for paragraphs by given texts and updates their formatting, following updates are supported:
        :param texts: list of texts used for searching the paragraphs
        :param named_style: used for changing paragraphs to heading-type
        :param line_spacing: used for changing paragraph's line-spacing
        :param search_index_start: if set, all paragraphs before this index are skipped
        :param skip_formatted_paragraphs: set to True to skip already formatted paragraphs
        """
        requests: dict[str, Any] = {'requests': []}
        for text in texts:
            if not (start_index := self._find_text_in_doc(
                    text, search_index_start=search_index_start, skip_formatted_paragraphs=skip_formatted_paragraphs)):
                LOG.warning(f'String {text} isn\'t present in current document. Skipping! (No changes were made)')
                continue
            start_index = start_index[0]
            request: dict[str, Any] = {'updateParagraphStyle': {
                'range': {
                    'startIndex': start_index, 'endIndex': start_index + len(text)}}}

            if not any((named_style, line_spacing)):
                LOG.warning('Update paragraph style request does not contain any changes. Skipping!')
                continue

            fields: list[str] = []
            request['updateParagraphStyle']['paragraphStyle'] = {}
            if named_style:
                request['updateParagraphStyle']['paragraphStyle']['namedStyleType'] = named_style
                fields.append('namedStyleType')
            if line_spacing:
                request['updateParagraphStyle']['paragraphStyle']['lineSpacing'] = line_spacing * 100
                fields.append('lineSpacing')
            request['updateParagraphStyle']['fields'] = ', '.join(fields)
            requests['requests'].append(request)
        if requests['requests']:
            if self._batch_requests:
                self._awaiting_requests['requests'] += requests['requests']
            else:
                self._execute_batch_update(requests)

    def update_all_paragraphs_style_request(self, line_spacing: float | None = None) -> None:
        """Currently only changing line spacing is supported for the whole document."""
        if line_spacing is None:
            LOG.warning('Currently only change in line spacing for whole document is available.')
            return
        if content := self._get_body_content():
            start_index = 1
            end_index = content[-1]['endIndex']
            request: dict[str, Any] = {'updateParagraphStyle': {
                'range': {
                    'startIndex': start_index, 'endIndex': end_index},
                'paragraphStyle': {
                    'lineSpacing': line_spacing * 100},
                'fields': 'lineSpacing'}}
            if self._batch_requests:
                self._awaiting_requests['requests'].append(request)
            else:
                self._execute_batch_update({'requests': [request]})
        else:
            LOG.warning('Document body is empty, therefore no change was performed.')

    def update_text_style_request(
            self, texts: list[str], *, text_style: str | None = None, heading_link: str | None = None,
            url_link: str | None = None, search_index_start: int | None = None, search_index_end: int | None = None,
            find_all: bool = False, find_one_paragraph: bool = False, skip_formatted_paragraphs: bool = False) -> None:
        """
        Searches for text strings and updates their formatting, following updates are supported:
        :param texts: list of text strings which should be formatted
        :param text_style: used for changing text style to ['bold', 'italic', 'underline', 'strikethrough' or
            'smallCaps']
        :param heading_link: ID of specific heading in the document used to create inter-document links for the texts
        :param url_link: URL address to link for the given texts
        :param find_all: set to True to find all non-overlapping occurrences of the given texts, otherwise only first
            occurrence is updated
        :param find_one_paragraph: set to True to find all occurrences within a single paragraph (mutually exclusive
            with find_all)
        :param search_index_start: if set, all paragraphs before this index are skipped
        :param search_index_end: if set, all paragraphs after this index are skipped
        :param skip_formatted_paragraphs: set to True to skip already formatted paragraphs
        """
        if find_all and find_one_paragraph:
            raise ValueError('find_all and find_one_paragraph are mutually exclusive')
        if not any((text_style, heading_link, url_link)):
            raise ValueError('text_style, heading_link or url_link must be set')
        elif text_style:
            text_style_request = {f'{text_style}': True}
            field = f'{text_style}'
        elif heading_link:
            text_style_request = {'link': {'headingId': f'{heading_link}'}}
            field = 'link'
        else:
            text_style_request = {'link': {'url': f'{url_link}'}}
            field = 'link'
        requests: dict[str, Any] = {'requests': []}
        for text in texts:
            if not (start_indexes := self._find_text_in_doc(
                    text, search_index_start=search_index_start, search_index_end=search_index_end, find_all=find_all,
                    skip_formatted_paragraphs=skip_formatted_paragraphs)):
                LOG.warning(f'Text {text} was not found in the document (or specified range), no style updates were '
                            f'made.')
                continue
            if not find_all and not find_one_paragraph:
                start_indexes = [start_indexes[0]]
            for start_index in start_indexes:
                request: dict[str, Any] = {'updateTextStyle': {
                    'range': {'startIndex': start_index, 'endIndex': start_index + len(text)},
                    'textStyle': text_style_request,
                    'fields': field
                }}
                requests['requests'].append(request)
        if requests['requests']:
            if self._batch_requests:
                self._awaiting_requests['requests'] += requests['requests']
            else:
                self._execute_batch_update(requests)

    def update_table_style_request(self, table_idx: int, line_spacing: float | None = None) -> None:
        """Currently only changing line spacing is supported for the whole table. Table is specified using table index
         (0-based index of tables in the document, starting from the top of the document)."""
        if line_spacing is None:
            LOG.warning('Currently only change in line spacing for whole table is available.')
            return
        table_range = self.get_ith_table_doc_range(table_idx=table_idx)
        request: dict[str, Any] = {'updateParagraphStyle': {
            'range': {
                'startIndex': table_range[0], 'endIndex': table_range[1]},
            'paragraphStyle': {
                'lineSpacing': line_spacing * 100},
            'fields': 'lineSpacing'}}
        if self._batch_requests:
            self._awaiting_requests['requests'].append(request)
        else:
            self._execute_batch_update({'requests': [request]})

    def update_table_cell_style_request(self, table_idx: int, remove_border: bool = False) -> None:
        """Currently only removing borders is supported for the whole table (cell style). Table is specified using table
        index (0-based index of tables in the document, starting from the top of the document)."""
        border_style = {
            'color': {'color': {'rgbColor': {'red': 0, 'green': 0, 'blue': 0}}},
            'dashStyle': 'SOLID',
            'width': {'magnitude': 0, 'unit': 'PT'},
        }

        table_start_idx = self.get_ith_table_doc_range(table_idx)[0]
        if remove_border:
            request = {
                'updateTableCellStyle': {
                    'tableStartLocation': {'index': table_start_idx},
                    'tableCellStyle': {
                        'borderTop': border_style,
                        'borderBottom': border_style,
                        'borderLeft': border_style,
                        'borderRight': border_style,
                    },
                    'fields': '*',
                }
            }
            if self._batch_requests:
                self._awaiting_requests['requests'].append(request)
            else:
                self._execute_batch_update({'requests': [request]})
        else:
            LOG.warning('Currently no other table cell styling than removing border is supported, no changes were '
                        'applied.')

    def update_table_column_properties_request(self, table_idx: int, column_widths: list[float]) -> None:
        """Currently only changing column widths is supported. Table is specified using table index (0-based index of
        tables in the document, starting from the top of the document)."""
        table = self._get_ith_table_of_document(table_idx)
        if dictutil.getValue(table, 'table.columns', -1) != len(column_widths):
            raise IndexError(f'Column widths {column_widths} are out of range, for table with'
                             f' {dictutil.getValue(table, "table.columns")} columns!')
        requests: list[dict] = []
        for column_index, width_in_inches in enumerate(column_widths):
            width_points = round(width_in_inches * 72, 3)
            requests.append({
                'updateTableColumnProperties': {
                    'tableStartLocation': {'index': table['startIndex']},
                    'columnIndices': [column_index],
                    'tableColumnProperties': {
                        'widthType': 'FIXED_WIDTH',
                        'width': {'magnitude': width_points, 'unit': 'PT'},
                    },
                    'fields': '*',
                }
            })
        if self._batch_requests:
            self._awaiting_requests['requests'] += requests
        else:
            self._execute_batch_update({'requests': requests})
