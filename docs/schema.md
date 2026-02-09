# 输出规范 / Output Schema

> 返回 [README](../README.md)

完整规范见 `agent/schema.json`，核心结构：

```json
{
  "generated_at": "2026-02-07T14:22:03",
  "summary": {
    "total_datasets": 14,
    "total_github_orgs": 14,
    "total_github_repos": 136,
    "total_github_repos_high_relevance": 80,
    "total_papers": 22,
    "total_blog_posts": 93,
    "total_x_tweets": 47,
    "total_trending_datasets": 5
  },
  "datasets": [{
    "id": "allenai/Dolci-Instruct-SFT",
    "category": "sft_instruction",
    "created_at": "2025-11-18T00:00:00.000Z",
    "last_modified": "2026-02-03T12:34:56.000Z",
    "downloads": 2610,
    "growth_7d": 0.35,
    "growth_30d": 1.2,
    "languages": ["en", "zh"],
    "license": "odc-by"
  }],
  "github_activity": [{
    "org": "openai",
    "repos_count": 12,
    "repos_updated": [{
      "name": "open-instruct",
      "full_name": "openai/open-instruct",
      "stars": 1500,
      "relevance": "high",
      "relevance_signals": ["dataset", "instruction"]
    }]
  }],
  "papers": [{
    "title": "...",
    "created_at": "2026-02-04T16:53:47",
    "source": "arxiv",
    "is_dataset_paper": true
  }],
  "blog_posts": [{
    "source": "OpenAI Blog",
    "articles": [{"title": "...", "url": "...", "date": "2026-02-05", "summary": "..."}]
  }],
  "x_activity": {
    "accounts": [{
      "username": "karpathy",
      "relevant_tweets": [{"text": "...", "url": "...", "date": "2026-02-06"}]
    }]
  },
  "reddit_activity": {
    "posts": [{
      "subreddit": "MachineLearning",
      "title": "New RLHF dataset released",
      "url": "https://reddit.com/r/MachineLearning/...",
      "score": 150,
      "signals": ["rlhf", "dataset"]
    }],
    "metadata": {"subreddits_checked": 5, "total_posts": 42}
  },
  "competitor_matrix": {
    "matrix": {"openai": {"sft": 2, "preference": 1, "repos": 5}},
    "top_orgs": [{"org": "openai", "total": 13}]
  },
  "dataset_lineage": {
    "edges": [{"source": "openai/gsm8k", "target": "meta/gsm8k-v2", "type": "derived"}],
    "version_chains": {"gsm8k": ["v1", "v2", "v3"]},
    "fork_trees": {"alpaca": ["alpaca-cleaned", "alpaca-gpt4"]},
    "root_datasets": [{"id": "openai/gsm8k", "count": 5}]
  },
  "org_graph": {
    "nodes": ["openai", "meta", "google"],
    "edges": [{"source": "openai", "target": "meta", "type": "shared_topic", "weight": 3}],
    "clusters": [["openai", "meta", "google"]],
    "centrality": {"openai": 0.667, "meta": 0.333}
  },
  "alerts": [
    {
      "rule": "zero_data_github",
      "severity": "critical",
      "title": "GitHub: 0 active orgs",
      "detail": "Data source 'github' returned 0 results. Check connectivity and API keys.",
      "timestamp": "2026-02-09T08:00:00",
      "fingerprint": "a1b2c3d4e5f67890"
    }
  ]
}
```
