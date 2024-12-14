# Agentic Copilot

**Agentic Copilot** is a multi-agent system designed with the LlamaIndex framework to create a resource-expert chatbot. This project was developed as part of my University Bachelor's thesis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To set up the environment and install the dependencies for this project, use **Poetry**:

1. Install Poetry (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install the project dependencies:

   ```bash
   poetry install
   ```

## Usage

Once the environment is set up, you can start the application:

1. Navigate to the src directory:

   ```bash
   cd src
   ```

2. Run the application:

   ```bash
   python -m main
   ```

## Requirements

The application requires a LiteLLM deployment that is listening on http://0.0.0.0:4000.

You can learn more about LiteLLM and how to set up a proxy server by visiting the official LiteLLM page: [LiteLLM Documentations](https://docs.litellm.ai/).

## Configuration

You must create a local.yaml file under the the config directory. The local.yaml file should contain the following configuration:

```yaml
# Logging level: Controls the level of logs generated
log_level: info  # Options: debug, info, warning, error, critical
# Azure endpoint configuration
azure_endpoint: ***  # Replace with your Azure endpoint URL
# Azure API key for authentication
azure_api_key: ***  # Replace with your Azure API key
# Temperature setting for model response (higher = more random, lower = more focused)
azure_temperature: ***  # Example: 0.7 (typically between 0 and 1)
# Model name used for embeddings
embedding_model: ***  # Specify the embedding model name
# Deployment name for the embedding model
embedding_deployment: ***  # Specify the deployment name
# API version to use for embedding model
embedding_api_version: ***  # Example: '2023-05-15'
# Maximum number of function calls allowed
max_function_calls: ***  # Set a limit to prevent excessive API calls
```

## Project Structure

This is the directory structure of the project:

```plainetxt
├── config/                # Configuration files for the project
│   └── local.yaml         # Local configuration file
├── src/                   # Main source code directory
│   ├── agentic_copilot/   # Core application folder
│   │   ├── config/        # Configuration for agentic_copilot
│   │   ├── models/        # Models for the application
│   │   │   ├── agents/    # Agent and tool implementations
│   │   │   └── utils/     # Multi-Agent framework
│   │   └── workflows/     # Workflow-related logic
│   ├── data/              # Data files
│   ├── main.py            # Entry point for the application
│   └── test.ipynb         # Jupyter notebook I used for testing
└── pyproject.toml         # Poetry package manager configuration
```
