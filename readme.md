# baremetal-ai

This repository illustrates how to use RAG (Retrieval Augmented Generation) to create and use a model for in this case baremetal Raspberry Pi development.

It uses the following:

- Python 3.12
- Ollama

## Ollama

Ollama is a command-line tool that allows you to run large language models locally. It provides a simple interface for downloading and running models.

Install Ollama by following the instructions on their [website](https://ollama.com/).

We use the following local models:

- llama3.1:8b
- nomic-embed-text

Install these models by running the following commands:

```bash
ollama run llama3.1:8b
ollama run nomic-embed-text
```

You can see which models you are running by executing the following command:

```bash
ollama list
```

Models are exposed as REST API's on localhost:11434. The scripts use the OpenAPI library to interact with the models.

Install the Python OpenAI API by running the following command:

```bash
pip install openai
```

## Miscellaneous

The scripts uses the following libraries:

- pandas
- numpy
- sklearn
- langchain-text-splitters (for splitting markdown files and creating controlled chunks)

Install them as follows:

```bash
pip install pandas numpy scikit-learn
```

## Usage

To generate a model, run the applications as:

```
python main.py build_model --log-dir logs --content-dir <content-dir> --model model.csv
```

To use the model, run the applications as:

```
python main.py chatbot --log-dir logs --model model.csv
```

Enter a question, and wait for the answer.
