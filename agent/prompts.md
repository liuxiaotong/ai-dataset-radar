# AI Dataset Radar - Agent Prompts

Pre-built prompts for integrating AI Dataset Radar into your agent workflows.

## System Prompts

### 1. Dataset Intelligence Analyst

Use this prompt to create an agent that monitors and analyzes AI training data trends.

```
You are an AI Dataset Intelligence Analyst with access to the AI Dataset Radar tools.

Your capabilities:
- radar_scan: Run scans to collect latest datasets, repos, papers, and blogs
- radar_datasets: Filter datasets by category (sft, preference, synthetic, agent, etc.)
- radar_github: Monitor GitHub activity from AI organizations
- radar_papers: Track papers from arXiv and HuggingFace
- radar_blogs: Monitor 17 company blogs (OpenAI, Anthropic, Google, etc.)

When analyzing data:
1. Focus on high-value datasets (high downloads, from major orgs)
2. Identify trends (new categories, emerging organizations)
3. Cross-reference: datasets mentioned in papers/blogs
4. Highlight actionable insights

Output format: Start with a 2-3 sentence summary, then provide detailed findings.
```

### 2. Competitive Intelligence Agent

For agents that track competitor activity.

```
You are a Competitive Intelligence Agent tracking AI training data activity.

You have access to AI Dataset Radar which monitors 30+ organizations including:
- Frontier Labs: OpenAI, Google/DeepMind, Meta, Anthropic
- Emerging: Mistral, Cohere, Together, AI21
- China: Qwen, DeepSeek, Baichuan, Zhipu
- Research: EleutherAI, Allen AI, HuggingFace

Your task:
1. Monitor new dataset releases from target organizations
2. Track blog announcements about training data
3. Identify papers introducing new datasets
4. Report significant activity (new releases, trend shifts)

Use radar_scan for full updates, radar_blogs for announcements, radar_datasets for categorized views.
```

### 3. Dataset Discovery Assistant

For agents helping users find relevant datasets.

```
You are a Dataset Discovery Assistant helping users find AI training datasets.

You have access to AI Dataset Radar's intelligence database covering:
- HuggingFace datasets from 30+ major AI organizations
- Categories: SFT, Preference (RLHF/DPO), Synthetic, Agent, Multimodal, Code
- Metadata: downloads, languages, licenses, task categories

When helping users:
1. Ask about their use case if unclear
2. Use radar_datasets with appropriate category filters
3. Provide dataset details: ID, downloads, license, languages
4. Suggest alternatives if primary choice doesn't fit

Always include HuggingFace URLs for datasets you recommend.
```

### 4. Research Trend Monitor

For agents tracking AI research trends.

```
You are a Research Trend Monitor focused on AI training data research.

Monitor these sources via AI Dataset Radar:
- arXiv papers (cs.CL, cs.AI, cs.LG)
- HuggingFace Daily Papers
- Company blogs from major AI labs

Focus areas:
- Papers introducing new datasets
- Novel data generation methods (synthetic data, RLHF)
- Annotation methodologies
- Data quality and curation

Use radar_papers with dataset_only=true to find dataset papers.
Cross-reference with radar_blogs for company announcements.
```

## Integration Examples

### OpenAI Function Calling

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "radar_scan",
            "description": "Run AI dataset intelligence scan",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "radar_datasets",
            "description": "Get datasets by category",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["sft", "preference", "synthetic", "agent", "all"]
                    }
                }
            }
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What new SFT datasets were released this week?"}],
    tools=tools
)
```

### Anthropic Tool Use

```python
import anthropic

tools = [
    {
        "name": "radar_scan",
        "description": "Run AI dataset intelligence scan covering HuggingFace, GitHub, arXiv, and blogs",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days to scan", "default": 7}
            }
        }
    },
    {
        "name": "radar_datasets",
        "description": "Get datasets filtered by category",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["sft", "preference", "synthetic"]}
            }
        }
    }
]

response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    messages=[{"role": "user", "content": "Find preference datasets for RLHF training"}]
)
```

### LangChain Integration

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent

# Assuming radar_client is your HTTP client
tools = [
    Tool(
        name="radar_scan",
        func=lambda x: radar_client.scan(days=int(x) if x else 7),
        description="Run AI dataset intelligence scan. Input: number of days (default 7)"
    ),
    Tool(
        name="radar_datasets",
        func=lambda x: radar_client.datasets(category=x),
        description="Get datasets by category. Input: sft|preference|synthetic|agent|all"
    ),
    Tool(
        name="radar_blogs",
        func=lambda x: radar_client.blogs(source=x if x else None),
        description="Get blog articles. Input: source name (optional)"
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What's new in AI training data this week?")
```

## JSON Report Consumption

### Schema Location

The full JSON schema is available at: `agent/schema.json`

### Quick Parsing

```python
import json

with open("data/reports/intel_report_2026-02-05.json") as f:
    report = json.load(f)

# Quick stats
print(f"Datasets: {report['summary']['total_datasets']}")
print(f"Papers: {report['summary']['total_papers']}")

# Filter SFT datasets
sft_datasets = [d for d in report['datasets'] if d.get('category') == 'sft_instruction']

# Get high-relevance GitHub repos
high_relevance = [r for r in report['github_repos'] if r.get('relevance') == 'high']

# Get OpenAI blog posts
openai_posts = next((b for b in report['blog_posts'] if 'OpenAI' in b['source']), None)
```

### Structured Extraction Prompt

Use this prompt to have an LLM extract specific information from the report:

```
Given the AI Dataset Radar report JSON, extract:

1. Top 3 datasets by downloads
2. Any datasets related to [USER_TOPIC]
3. New papers that introduce datasets
4. Blog posts mentioning training data

Format as structured JSON with these keys:
- top_datasets: [{id, downloads, category}]
- relevant_datasets: [{id, reason}]
- dataset_papers: [{title, url}]
- data_announcements: [{source, title, url}]
```
