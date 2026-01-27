# JEDI

**Journalistic Excellence through Domain-specific Intelligence**

AI-powered transcript processing that cleans up interview transcripts and converts them into polished article format.

The JEDI Project is a collaboration between [Atex](https://www.atex.com/) and [Geneea](https://geneea.com).

---

[![PyPI version](https://img.shields.io/pypi/v/jedireporter)](https://pypi.org/project/jedireporter/)
[![Python Versions](https://img.shields.io/pypi/pyversions/jedireporter)](https://pypi.org/project/jedireporter/)
[![License](https://img.shields.io/pypi/l/jedireporter)](https://github.com/Geneea/jedireporter/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/jedireporter)](https://pypi.org/project/jedireporter/)
[![Build Status](https://github.com/Geneea/jedireporter/workflows/CI/badge.svg)](https://github.com/Geneea/jedireporter/actions)

## Installation

```bash
pip install -U jedireporter
```

## Quick Start

```bash
# Convert plain text interview to transcript JSON
jedireporter-impex import --format plain-text -i interview.txt -o transcript.jsonl

# Process transcript through the full pipeline
jedireporter -i transcript.jsonl -o article.jsonl --llm-profile default

# Export article to markdown
jedireporter-impex export --format markdown -i article.jsonl -o output.md
```

## Configuration

### Environment Variables

Copy the template file and fill in your credentials:

```bash
cp .env_template .env
# Edit .env with your API keys and credentials
```

The `jedireporter` command automatically loads `.env` from the current directory. Alternatively, set the environment variables directly in your shell.

### LLM Profiles

Select an LLM profile using `--llm-profile <name>`. Available profiles:

| Profile | Provider | Model | Environment Variable |
|---------|----------|-------|---------------------|
| default | OpenAI | gpt-5.1-2025-11-13 | `JEDI_OPENAI_API_KEY` |
| gpt-4.1 | OpenAI | gpt-4.1-2025-04-14 | `JEDI_OPENAI_API_KEY` |
| gpt-5 | OpenAI | gpt-5-2025-08-07 | `JEDI_OPENAI_API_KEY` |
| gpt-5-mini | OpenAI | gpt-5-mini-2025-08-07 | `JEDI_OPENAI_API_KEY` |
| gpt-5.1 | OpenAI | gpt-5.1-2025-11-13 | `JEDI_OPENAI_API_KEY` |
| gpt-5.2 | OpenAI | gpt-5.2-2025-12-11 | `JEDI_OPENAI_API_KEY` |
| claude | AWS Bedrock | Claude Sonnet 4.5 | AWS credentials (see below) |
| mistral | Mistral | mistral-large-2512 | `JEDI_MISTRAL_API_KEY` |

### AWS Bedrock Credentials (for `claude` profile)

```bash
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export AWS_DEFAULT_REGION=<aws-region>
export AWS_BEDROCK_ROLE=<bedrock-role-arn>
```

### Langfuse Observability (Optional)

To enable [Langfuse](https://langfuse.com/) tracing for LLM calls, set both environment variables:

```bash
export LANGFUSE_PUBLIC_KEY=<your-public-key>
export LANGFUSE_SECRET_KEY=<your-secret-key>
```

When not configured, the pipeline runs without tracing and no warnings are shown.

## Transcript Processing Workflow

The main `jedireporter` command processes transcripts through a three-stage pipeline:

1. **Fix Transcript** - Clean up ASR output, fix speaker boundaries, remove fillers
2. **Create Article** - Convert interview into structured article with topics, Q&A, and summary
3. **Post-Process Article** - Add styling, enrich with web search results

### Usage

```bash
jedireporter -i transcript.jsonl -o article.jsonl --llm-profile <profile>
```

### Options

| Option | Description |
|--------|-------------|
| `-i`, `--input-file` | Path to input transcript JSON file |
| `-o`, `--output-file` | Path to output article JSON file |
| `--llm-profile` | LLM profile name (default: `default`) |
| `--include-transcript` | Include transcript in the article output |
| `--max-concurrency` | Maximum parallel workflow invocations (default: 4) |

## Input Format

Transcript JSON structure (one JSON object per line in `.jsonl` format).

### Minimal Example (mandatory fields only)

```json
{
  "id": "interview-001",
  "language": "en",
  "segments": [
    {"id": "0", "speaker_id": "spk_0", "text": "Welcome to the show."},
    {"id": "1", "speaker_id": "spk_1", "text": "Thank you for having me."}
  ]
}
```

### Full Example (with optional fields)

```json
{
  "id": "interview-001",
  "language": "en",
  "segments": [
    {
      "id": "0",
      "speaker_id": "spk_0",
      "text": "Welcome to the show.",
      "timecodes": {"start_time": 0.0, "end_time": 2.5}
    },
    {
      "id": "1",
      "speaker_id": "spk_1",
      "text": "Thank you for having me.",
      "timecodes": {"start_time": 2.5, "end_time": 4.0}
    }
  ],
  "speakers": [
    {"speaker_id": "spk_0", "role": "host", "name": "John Smith", "description": "Host of the Morning Show"},
    {"speaker_id": "spk_1", "role": "guest", "name": "Jane Doe", "description": "CEO of TechCorp"}
  ],
  "url": "https://youtube.com/watch?v=abc123"
}
```

### Mandatory Fields

| Field | Description |
|-------|-------------|
| `id` | Unique transcript identifier |
| `language` | Language code (e.g., "en", "cs") |
| `segments` | List of dialogue segments |
| `segments[].id` | Unique segment identifier |
| `segments[].speaker_id` | Speaker identifier |
| `segments[].text` | Spoken text content |

### Optional Fields

| Field | Description |
|-------|-------------|
| `segments[].timecodes` | Start and end times in seconds |
| `speakers` | Speaker metadata list |
| `speakers[].speaker_id` | Matches segment speaker_id (required if speakers provided) |
| `speakers[].role` | "host", "guest", or "other" (required if speakers provided) |
| `speakers[].name` | Human-readable name |
| `speakers[].description` | Title, affiliation, etc. |
| `url` | Source audio/video URL (enables timecode linking in exports)

## Import/Export (impex)

### Import: Plain Text to Transcript JSON

Convert plain text interviews to transcript JSON format:

```bash
jedireporter-impex import --format plain-text -i interview.txt -o transcript.jsonl
```

**Plain text format:**

```
Speaker Name: What they said in this segment...
Another Speaker: Their response goes here...
Speaker Name: And the conversation continues...
```

Each line follows the pattern `Speaker Name: Text content`. Empty lines are ignored.

### Import: AWS Transcribe JSON to Transcript JSON

Convert [AWS Transcribe](https://docs.aws.amazon.com/transcribe/latest/dg/how-it-works.html) output to transcript JSON format:

```bash
jedireporter-impex import --format aws-transcribe -i transcribe-output.json -o transcript.jsonl
```

**Note:** Speaker diarization must be enabled in AWS Transcribe for speaker labels to be included. Without diarization, all segments will be assigned to `spk_0`.

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Input format: `plain-text` or `aws-transcribe` |
| `--language` | Language code for the transcript |
| `-i`, `--input-file` | Input file path (supports glob patterns) |
| `-o`, `--output-file` | Output file path |

### Export: Article JSON to Markdown

```bash
jedireporter-impex export --format markdown -i article.jsonl -o output.md
```

If output is a directory, each article is saved as a separate file.

### Export: Article JSON to Google Docs

```bash
jedireporter-impex export --format google-doc -i article.jsonl --credentials service-account.json
```

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Output format: `markdown` or `google-doc` |
| `--credentials` | Path to Google service account JSON file |
| `--token` | Path to OAuth access token file |
| `--google-folder` | Google Drive folder ID for output |
| `-i`, `--input-file` | Input article JSON file |
| `-o`, `--output-file` | Output file path (markdown only) |

## REST API Service

Run the transcript processing pipeline as a web service.

### Starting the Server

```bash
jedireporter-server --bind localhost --port 5000
```

| Option | Description |
|--------|-------------|
| `--bind` | Network interface to listen on (default: `localhost`) |
| `--port` | Port number (default: `5000`) |

### Configuration

The server uses the same environment variables as the CLI:

- `JEDI_LLM_PROFILE` - LLM profile name (default: `default`)
- `JEDI_SERVER_PATH` - URL path prefix for reverse proxy setups

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Health check |
| `/v1/generate-article` | POST | Generate article from transcript |

### Example: Generate Article

```bash
curl -X POST http://localhost:5000/v1/generate-article \
  -H "Content-Type: application/json" \
  -d '{
    "id": "interview-001",
    "language": "en",
    "segments": [
      {"id": "0", "speakerId": "host", "text": "Welcome to the show."},
      {"id": "1", "speakerId": "guest", "text": "Thank you for having me."}
    ],
    "speakers": [
      {"speakerId": "host", "role": "host", "name": "John Smith"},
      {"speakerId": "guest", "role": "guest", "name": "Jane Doe"}
    ]
  }'
```

The API uses camelCase for JSON field names. See `/docs` for interactive API documentation (Swagger UI).

## Command Reference

| Command | Description |
|---------|-------------|
| `jedireporter` | Main processing workflow (transcript to article) |
| `jedireporter-server` | Start the REST API service |
| `jedireporter-impex import` | Convert plain text or AWS Transcribe output to transcript JSON |
| `jedireporter-impex export` | Convert article JSON to markdown/Google Docs |

## Example End-to-End Workflow

```bash
# 1. Set up API key
export JEDI_OPENAI_API_KEY=sk-...

# 2. Prepare plain text interview file (interview.txt)
cat > interview.txt << 'EOF'
Host: Welcome to today's show. We have a special guest joining us.
Guest: Thank you for having me. I'm excited to be here.
Host: Let's dive right in. What inspired your latest project?
Guest: It all started when I noticed a gap in the market...
EOF

# 3. Import to transcript JSON
jedireporter-impex import --format plain-text --language en -i interview.txt -o transcript.jsonl

# 4. Process through the pipeline
jedireporter -i transcript.jsonl -o article.jsonl --llm-profile default

# 5. Export to markdown
jedireporter-impex export --format markdown -i article.jsonl -o article.md

# 6. View result
cat article.md
```

## Funding

This project is supported by [NextGenerationEU](https://next-generation-eu.europa.eu/index_en), the European Union's recovery and resilience initiative, through the Recovery and Resilience Facility.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
