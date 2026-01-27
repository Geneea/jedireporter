# JEDI

**Journalistic Excellence through Domain-specific Intelligence**

AI-powered transcript processing that cleans up interview transcripts and converts them into polished article format.

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

Transcript JSON structure (one JSON object per line in `.jsonl` format):

```json
{
  "id": "interview-001",
  "language": "en",
  "segments": [
    {
      "id": "0",
      "speaker_id": "spk_0",
      "text": "Welcome to the show.",
      "timecodes": {
        "start_time": 0.0,
        "end_time": 2.5
      }
    },
    {
      "id": "1",
      "speaker_id": "spk_1",
      "text": "Thank you for having me.",
      "timecodes": {
        "start_time": 2.5,
        "end_time": 4.0
      }
    }
  ],
  "speakers": [
    {
      "speaker_id": "spk_0",
      "role": "host",
      "name": "John Smith",
      "description": "Host of the Morning Show"
    },
    {
      "speaker_id": "spk_1",
      "role": "guest",
      "name": "Jane Doe",
      "description": "CEO of TechCorp"
    }
  ],
  "url": "https://youtube.com/watch?v=abc123"
}
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique transcript identifier |
| `language` | Yes | Language code (e.g., "en", "cs") |
| `segments` | Yes | List of dialogue segments |
| `segments[].id` | Yes | Unique segment identifier |
| `segments[].speaker_id` | Yes | Speaker identifier |
| `segments[].text` | Yes | Spoken text content |
| `segments[].timecodes` | No | Start and end times in seconds |
| `speakers` | No | Speaker metadata |
| `speakers[].speaker_id` | Yes | Matches segment speaker_id |
| `speakers[].role` | Yes | "host", "guest", or "other" |
| `speakers[].name` | No | Human-readable name |
| `speakers[].description` | No | Title, affiliation, etc. |
| `url` | No | Source audio/video URL |

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

**Options:**

| Option | Description |
|--------|-------------|
| `--format` | Input format (currently: `plain-text`) |
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

## Command Reference

| Command | Description |
|---------|-------------|
| `jedireporter` | Main processing workflow (transcript to article) |
| `jedireporter-impex import` | Convert plain text to transcript JSON |
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
