# 🔭 AI Dataset Radar

Daily tracker for new AI datasets and benchmarks. Never miss important releases from Hugging Face, Papers with Code, and arXiv.

## ✨ Features

- **Multi-source Tracking**: Monitors Hugging Face Datasets, Papers with Code benchmarks, and arXiv dataset papers
- **Smart Filtering**: Filters by downloads, stars, domain keywords, and trending velocity
- **Daily Updates**: Automated via GitHub Actions, runs every day at 8:00 AM UTC
- **Flexible Notifications**: Email digest, Webhook, or RSS feed

## 📊 Data Sources

| Source | What it tracks | Update frequency |
|--------|---------------|------------------|
| 🤗 Hugging Face | New datasets | Daily |
| 📈 Papers with Code | New benchmarks & SOTA | Daily |
| 📄 arXiv | Dataset papers (cs.CL, cs.CV, cs.LG) | Daily |

## 🚀 Quick Start

### 1. Fork this repo

### 2. Configure your filters

Edit `config.yaml`:
     
      - ```yaml
        filters:
          min_downloads: 100
          min_stars: 10
          domains:
            - code
            - agent
            - reasoning
            - multimodal

        notifications:
          email: your@email.com
          # webhook: https://your-webhook-url
        ```

        ### 3. Enable GitHub Actions

        Go to Settings → Actions → Enable workflows

        ## 📁 Project Structure

        ```
        ai-dataset-radar/
        ├── src/
        │   ├── scrapers/           # Data source scrapers
        │   │   ├── huggingface.py
        │   │   ├── paperswithcode.py
        │   │   └── arxiv.py
        │   ├── filters.py          # Filtering logic
        │   └── notifiers.py        # Notification handlers
        ├── data/                   # Daily snapshots (JSON)
        ├── .github/workflows/      # GitHub Actions
        ├── config.yaml             # Your configuration
        └── requirements.txt
        ```

        ## 📬 Output Example

        Each daily run generates a report like:

        ```
        🆕 New Datasets (2025-01-29)

        🤗 Hugging Face:
          - microsoft/phi-4-code-instruct (↑1.2k downloads)
          - allenai/tulu-3-eval-suite (Code evaluation)

        📈 Papers with Code:
          - AgentBench v2.0 (Agent evaluation benchmark)

        📄 arXiv:
          - "MEGA-Bench: Scaling Multimodal Evaluation" (2501.xxxxx)
        ```

        ## 🛠️ Development

        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Run manually
        python src/main.py

        # Run tests
        pytest tests/
        ```

        ## 📄 License

        MIT License - feel free to use and modify.

        ---

        Made with ❤️ for the AI research community
