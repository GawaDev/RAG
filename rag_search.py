# rag_search.py

from typing import Any, Dict, List

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity

# --- 設定 ---
# EMBEDDING_MODEL = "nomic-embed-text" # RAGベクトル生成用モデル (app.pyから受け取る)
TOP_K = 3  # 検索するチャンクの数


# --- ヘルパー関数 ---
def find_similar_chunks(
    query_vector: List[float], chunks_with_vectors: List[Dict[str, Any]], top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    クエリベクトルとチャンクのベクトルを比較し、類似度の高い上位K個のチャンクを返す。
    """
    if not chunks_with_vectors:
        return []

    chunk_vectors = np.array([chunk["vector"] for chunk in chunks_with_vectors])
    query_vector_np = np.array(query_vector).reshape(1, -1)

    # コサイン類似度を計算
    similarities = cosine_similarity(query_vector_np, chunk_vectors)[0]

    # 類似度の高い順にインデックスを取得
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # 該当するチャンクを返す
    retrieved_chunks = [chunks_with_vectors[i] for i in top_k_indices]
    return retrieved_chunks


def format_rag_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    取得したチャンクをLLMに渡すための整形されたコンテキスト文字列に変換する。
    """
    if not retrieved_chunks:
        return "【参照情報】\n関連情報が見つかりませんでした。"

    context_str = "【参照情報】\n"
    for i, chunk in enumerate(retrieved_chunks):
        context_str += "----------\n"
        context_str += f"チャンク: {i + 1}\n"
        context_str += f"ファイル: {chunk.get('source_file', '不明')}\n"
        context_str += f"内容:\n{chunk['text']}\n"
    context_str += "----------\n"
    return context_str


# --- メイン処理関数 (RAG検索とコンテキスト取得) ---
def get_rag_context(
    query: str, processed_rag_data: List[Dict[str, Any]], ollama_client: ollama.Client
) -> str:
    """
    ユーザーのクエリに対してRAG検索を実行し、LLMに渡すコンテキストを返す。
    """
    if not processed_rag_data:
        return "【参照情報】\nRAGデータがありません。"

    query_vector = None
    try:
        # クエリのベクトル埋め込みを生成 (データ前処理と同じモデルを使用)
        # EMBEDDING_MODEL は app.py から渡されるか、ここで別途定義
        # app.py で RAG_EMBEDDING_MODEL を使用しているため、ここでの定義は不要
        response = ollama_client.embeddings(
            model="nomic-embed-text",  # app.pyと合わせる
            prompt=query,
        )
        query_vector = response["embedding"]
    except Exception as e:
        print(f"Error generating query embedding for RAG: {e}")
        return f"【参照情報】\nクエリのベクトル化に失敗しました: {e}"

    # 関連チャンクを検索
    retrieved_chunks = find_similar_chunks(query_vector, processed_rag_data, TOP_K)

    # 取得したチャンクをコンテキスト形式に整形
    context_content = format_rag_context(retrieved_chunks)

    # 修正: 整形されたコンテキスト文字列を返す
    return context_content


# --- スクリプトとして直接実行する部分 ---
if __name__ == "__main__":
    # このブロックはテスト用です。
    # app.py は get_rag_context 関数を直接呼び出します。

    # テスト用の設定
    OLLAMA_HOST_TEST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL_TEST = "nomic-embed-text"
    TEST_RAG_CACHE_FILE = (
        "./test_processed_rag_data_cache.json"  # テスト用のキャッシュファイル
    )

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

    # テスト用RAGデータをロード（preprocess_data_rag.pyで生成したものなど）
    processed_test_data = []
    if os.path.exists(TEST_RAG_CACHE_FILE):
        with open(TEST_RAG_CACHE_FILE, "r", encoding="utf-8") as f:
            processed_test_data = json.load(f)
        print(f"Loaded {len(processed_test_data)} chunks from test cache.")
    else:
        print(
            f"Test RAG data cache not found at {TEST_RAG_CACHE_FILE}. Please run preprocess_data_rag.py first to generate data."
        )
        exit()

    test_query = "生成AIの利用ガイドラインについて教えてください。"
    print(f"\nTest Query: {test_query}")

    rag_context = get_rag_context(test_query, processed_test_data, test_ollama_client)
    print("\n--- Generated RAG Context ---")
    print(rag_context)
    print("-----------------------------")

    # ここではLLMでの回答生成は行いません。それはllm_generate.pyの役割です。
