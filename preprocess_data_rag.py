import json
import os
from typing import Any, Dict, List, Optional

import ollama

# --- 設定 ---
# 環境変数または直接設定
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# EMBEDDING_MODEL = "nomic-embed-text" # RAGベクトル生成用モデル
# DATA_DIR = "./data/raw" # データディレクトリ

# app.pyから直接引数で受け取るため、ここでは定数として持たないか、デフォルト値としてのみ使う


# --- ヘルパー関数 ---
def simple_chunk_text(
    text: str, chunk_size: int = 500, overlap_size: int = 50
) -> List[str]:
    """
    シンプルなテキストチャンキング関数。
    """
    chunks = []
    if not text:
        return chunks

    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += chunk_size - overlap_size
        if start_index >= len(text):  # 最後のチャンクがチャンクサイズより小さい場合
            break  # 処理を終了
    return chunks


def read_file_content(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = ""

        if isinstance(data, list):  # ✅ リスト形式の対応追加
            for doc in data:
                if "title" in doc:
                    content += f"Title: {doc['title']}\n"
                if "sections" in doc:
                    for section in doc["sections"]:
                        if "section_title" in section:
                            content += f"\nHeading: {section['section_title']}\n"
                        if "content" in section and isinstance(
                            section["content"], list
                        ):
                            for item in section["content"]:
                                if "text" in item:
                                    content += item["text"] + "\n"

        elif isinstance(data, dict):  # 単一ドキュメント形式
            if "title" in data:
                content += f"Title: {data['title']}\n"
            if "sections" in data:
                for section in data["sections"]:
                    if "section_title" in section:
                        content += f"\nHeading: {section['section_title']}\n"
                    if "content" in section and isinstance(section["content"], list):
                        for item in section["content"]:
                            if "text" in item:
                                content += item["text"] + "\n"
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)

        return content.strip() if content else None

    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


# --- RAGデータ前処理のメイン関数 ---
def preprocess_for_rag(
    data_dir: str, embedding_model: str, ollama_client: ollama.Client
) -> List[Dict[str, Any]]:
    """
    指定されたディレクトリからファイルを読み込み、チャンクに分割し、ベクトル埋め込みを生成する。
    """
    all_chunks_with_vectors = []

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return []

    print(
        f"Starting RAG preprocessing from {data_dir} using model '{embedding_model}'..."
    )

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            print(f"Processing file: {filename}")
            content = read_file_content(filepath)
            if content:
                chunks = simple_chunk_text(content)
                for i, chunk_text in enumerate(chunks):
                    try:
                        # OllamaでEmbedding生成
                        response = ollama_client.embeddings(
                            model=embedding_model, prompt=chunk_text
                        )
                        chunk_vector = response["embedding"]

                        all_chunks_with_vectors.append(
                            {
                                "text": chunk_text,
                                "vector": chunk_vector,
                                "source_file": filename,
                                "chunk_id": i,
                            }
                        )
                    except Exception as e:
                        print(
                            f"  Error generating embedding for chunk {i} in {filename}: {e}"
                        )
            else:
                print(f"  Skipping empty or unreadable file: {filename}")

    print(
        f"--- RAG Preprocessing Complete. Total chunks processed: {len(all_chunks_with_vectors)} ---"
    )

    # 修正: 処理されたデータを返す
    return all_chunks_with_vectors


# --- スクリプトとして実行された場合の処理 (app.pyからは直接呼ばれない) ---
if __name__ == "__main__":
    # このブロック内の処理は、ファイルを直接実行したときのテストや初回処理用です。
    # app.py は上記の preprocess_for_rag 関数を直接呼び出します。

    # テスト用の設定（環境変数から読み込むか、直接指定）
    OLLAMA_HOST_TEST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL_TEST = "nomic-embed-text"
    DATA_DIR_TEST = "./data/raw"

    try:
        test_ollama_client = ollama.Client(host=OLLAMA_HOST_TEST)
        test_ollama_client.version()  # 接続確認
        print(
            f"Test Ollama client connected successfully. Version: {test_ollama_client.version()}"
        )
    except Exception as e:
        print(
            f"Test Ollama client connection failed: {e}. Ensure Ollama is running at {OLLAMA_HOST_TEST}"
        )
        exit()

    if not os.path.exists(DATA_DIR_TEST):
        print(
            f"Test Error: Data directory not found at {DATA_DIR_TEST}. Please create it and add files."
        )
        exit()

    # 処理を実行
    processed_rag_data_for_test = preprocess_for_rag(
        DATA_DIR_TEST, EMBEDDING_MODEL_TEST, test_ollama_client
    )

    # 結果の確認
    if processed_rag_data_for_test:
        print(
            f"\nTest run: Successfully processed {len(processed_rag_data_for_test)} chunks."
        )
        print(
            f"First chunk example (text): {processed_rag_data_for_test[0]['text'][:100]}..."
        )
        print(
            f"First chunk example (vector len): {len(processed_rag_data_for_test[0]['vector'])}"
        )

        # テストで保存する場合はここに記述
        # with open("test_processed_rag_data_cache.json", 'w', encoding='utf-8') as f:
        #     json.dump(processed_rag_data_for_test, f, ensure_ascii=False, indent=2)
        # print("Test RAG data saved to test_processed_rag_data_cache.json")
    else:
        print("Test run: RAG data processing failed or no chunks were generated.")
