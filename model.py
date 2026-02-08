from typing import List
import pandas as pd
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class Model:
    def __init__(self, client, logger):
        self.client = client
        self.logger = logger
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.temperature = 0.7

        self.model = pd.DataFrame()
    
    def load(self, path: str):
        self.logger.info(f"Loading model from {path}")
        embeddings = pd.read_csv(path)
        # Convert string representation of list to actual list
        embeddings["embedding"] = embeddings["embedding"].apply(json.loads)
        self.model = embeddings

    def save(self, path: str):
        self.logger.info(f"Saving model to {path}")
        self.model.to_csv(path, index=False)

    def ingest(self, directory: str) -> bool:
        self.logger.info(f"Extracting chunk data from files in {directory}")
        chunks = self.process_files(directory, chunk_size=500, chunk_overlap=50)
        if chunks == None:
            return False
        self.logger.info(f"Embedding data for {len(chunks)} chunks")
        self.model = self.generate_embeddings(chunks)
        return True

    def build_file_list(self, root_directory: str, scan_directory: str = None, exclude_directories: List[str] = [], include_extensions: List[str] = [], exclude_files: List[str] = []) -> List[str]:
        file_list = []
        if scan_directory is None:
            scan_directory = root_directory
        for item in os.listdir(scan_directory):
            if os.path.isdir(os.path.join(scan_directory, item)):
                if item in exclude_directories:
                    continue
                file_list.extend(self.build_file_list(root_directory, os.path.join(scan_directory, item), exclude_directories, include_extensions, exclude_files))
            else:
                rel_path = os.path.relpath(scan_directory, root_directory)
                base_name, ext = os.path.splitext(item)
                if ext == '':
                    if base_name.startswith('.'):
                        ext = base_name
                if ext in include_extensions:
                    if not item in exclude_files:
                        file_list.append(os.path.join(rel_path, item))
        return file_list

    def process_files(self, directory: str, chunk_size: int, chunk_overlap: int):
        file_chunks=[]
        file_list = self.build_file_list(directory, exclude_directories=['.git', '.vs', '.cmake', 'CMakeFiles'], include_extensions=['.md', '.txt'], exclude_files=['CMakeLists.txt', 'CMakeCache.txt', 'VSInheritEnvironments.txt'])
        for path in file_list:
            _, ext = os.path.splitext(path)
            if ext == '.txt':
                full_path = os.path.join(directory, path)
                with open(full_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    chunks = self.chunk_text(content, chunk_size, chunk_overlap)
                    for i in range(len(chunks)):
                        metadata = {'filename': path, 'chunk_id': i }
                        file_chunks.append({'metadata': metadata, 'content': chunks[i]})
            elif ext == '.md':
                # Create a chunking strategy with default configuration
                file_chunks.extend(self.process_markdown(directory, path, chunk_size, chunk_overlap))
        return file_chunks

    def process_markdown(self, root_directory: str, sub_path: str, chunk_size: int, chunk_overlap: int) -> List:
        file_chunks=[]

        full_path = os.path.join(root_directory, sub_path)
        with open(full_path, "r", encoding="utf-8") as file:
            content = file.read()

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
            md_header_splits = markdown_splitter.split_text(content)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Split
            splits = text_splitter.split_documents(md_header_splits)

            for doc in splits:
                metadata = doc.metadata
                metadata['filename'] = sub_path
                file_chunks.append({'metadata': metadata, 'content': doc.page_content})
        return file_chunks

    def generate_embeddings(self, chunks):
        embeddings = []
        index = 0
        count = len(chunks)
        for chunk in chunks:
            content = chunk['content']
            response = self.client.embeddings.create(
                model="nomic-embed-text",
                input=content
            )
            embeddings.append(response.data[0].embedding)
            index += 1
            percentage = int(index * 100 / count)
            print(f'Progress: {percentage}%', end='\r')

        result = pd.DataFrame(chunks)
        result['embedding'] = embeddings
        return result

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int):
        chunks = []
        words = text.split()  # Split by whitespace
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunks.append(' '.join(words[i: i+chunk_size]))
        return chunks

    def query(self, prompt: str):
        sorted_embeddings = self.find_similarities(prompt)

        total_count = len(self.model['embedding'])
        relevant_count = int(total_count / 10)
        # use top 5 most similar chunks as context
        top_chunks = sorted_embeddings.head(relevant_count)
        context = "\n\n".join(
            f"metadata: {row['metadata']}\n{row['content']}"
            for _, row in top_chunks.iterrows()
        )
        full_prompt = f"""
            Context:
            {context}

            User Query:
            {prompt}
            """
        self.logger.info('Thinking')
        response = self.client.chat.completions.create(
            model="llama3.1:8b",
            temperature=self.temperature,
            max_tokens=4000,
            messages=[self.system_message] +  [{"role": "user", "content": full_prompt}]
        )
        return response

    def find_similarities(self, prompt) -> pd.DataFrame:
        self.logger.info('Parsing question')
        response = self.client.embeddings.create(
            model="nomic-embed-text",
            input=prompt
        )
        input_embedding = response.data[0].embedding
        
        self.logger.info('Finding similarities')
        embeddings = self.model["embedding"].tolist()
        similarities = cosine_similarity([input_embedding], embeddings)

        self.model["similarity"] = similarities[0]
        sorted_embeddings = self.model.sort_values(by="similarity", ascending=False)

        return sorted_embeddings
