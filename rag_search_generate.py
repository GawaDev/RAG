# rag_search_generate.py

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import ollama
from sklearn.metrics.pairwise import (
    cosine_similarity,  # コサイン類似度計算のためにscikit-learnを使用
)

# --- 設定 ---
# RAGに使用するEmbeddingモデル名 (preprocess_data_rag.pyと合わせる)
EMBEDDING_MODEL = "nomic-embed-text"
# 回答生成に使用するLLMモデル名
LLM_MODEL = "mistral"  # 前回のやり取りで選択されたモデル
# 処理済みのRAGデータファイルパス
PROCESSED_DATA_FILE = "processed_rag_data.json"
# 検索で取得する関連チャンクの数
TOP_K = 3

# --- ヘルパー関数 ---


def load_processed_data(filepath: str) -> List[Dict[str, Any]]:
    """
    保存された処理済みRAGデータをファイルから読み込む。
    """
    if not os.path.exists(filepath):
        print(f"Error: Processed data file not found at {filepath}")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(
            f"Successfully loaded processed data from {filepath}. {len(data)} chunks found."
        )
        return data
    except Exception as e:
        print(f"Error loading processed data from {filepath}: {e}")
        return []


def find_similar_chunks(
    query_vector: List[float],
    chunks_with_vectors: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    クエリベクトルとチャンクベクトルの類似度を計算し、上位K個のチャンクを返す。
    """
    if not chunks_with_vectors:
        return []

    # クエリベクトルとチャンクベクトルをNumPy配列に変換
    query_vector_np = np.array(query_vector).reshape(
        1, -1
    )  # sklearnのために shape (1, N) にする
    # チャンクベクトルリストからNumPy配列を作成
    chunk_vectors_np = np.array([item["vector"] for item in chunks_with_vectors])

    # コサイン類似度を計算
    # cosine_similarity関数は (n_samples_x, n_features) と (n_samples_y, n_features) を取り、(n_samples_x, n_samples_y) の行列を返す
    similarities = cosine_similarity(query_vector_np, chunk_vectors_np)[
        0
    ]  # クエリは1つなので最初の行 [0] を取る

    # 類似度に基づいてチャンクのインデックスを取得
    # 類似度が高い順にソートし、上位K個のインデックスを取得
    sorted_indices = np.argsort(similarities)[::-1]  # 降順ソート
    top_k_indices = sorted_indices[:top_k]

    # 上位K個のチャンクを取得
    top_k_chunks = [chunks_with_vectors[i] for i in top_k_indices]

    # オプション：取得したチャンクと類似度を表示
    # print("\n--- Top K Similar Chunks ---")
    # for i, idx in enumerate(top_k_indices):
    #     print(f"Rank {i+1}: Source='{chunks_with_vectors[idx]['source']}', Chunk ID={chunks_with_vectors[idx]['chunk_id']}, Similarity={similarities[idx]:.4f}")
    #     print(f"  Text: \"{chunks_with_vectors[idx]['text'][:150]}...\"") # テキストの先頭150文字
    #     print("-" * 20)
    # print("--------------------------\n")

    return top_k_chunks


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    検索で取得したチャンクリストを、LLMへのプロンプトに含めるコンテキスト文字列に整形する。
    """
    if not retrieved_chunks:
        return "【参照情報】\n参照できる関連情報は見つかりませんでした。"

    context_string = "【参照情報】\n"
    for i, chunk in enumerate(retrieved_chunks):
        # 各チャンクの出典を明記
        source_info = f"文書: {chunk.get('source', '不明')}, チャンクID: {chunk.get('chunk_id', '不明')}"
        context_string += (
            f"<参照-{i + 1}>\n{source_info}\n{chunk['text']}\n</参照-{i + 1}>\n\n"
        )

    return context_string.strip()  # 末尾の不要な改行を削除


# --- メイン処理 ---


def get_rag_response(
    query: str, processed_rag_data: List[Dict[str, Any]]
) -> Optional[str]:
    """
    ユーザーのクエリに対してRAGを実行し、LLMの回答を生成する。
    """
    if not processed_rag_data:
        print("Error: No processed RAG data available.")
        return "現在、参照できる情報がありません。データ解析を実行してください。"

    ollama_client = None
    query_vector = None

    try:
        # Ollamaクライアント初期化
        ollama_client = ollama.Client()
        # クエリのベクトル埋め込みを生成 (データ前処理と同じモデルを使用)
        print(f"Generating embedding for query using '{EMBEDDING_MODEL}'...")
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_vector = response["embedding"]
        print("Query embedding generated.")

    except ollama.ResponseError as e:
        print(f"Error generating query embedding: {e}")
        return f"クエリのベクトル化に失敗しました: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during query embedding: {e}")
        return f"クエリ処理中にエラーが発生しました: {e}"

    # 関連チャンクを検索
    print(f"Searching for top {TOP_K} similar chunks...")
    retrieved_chunks = find_similar_chunks(query_vector, processed_rag_data, TOP_K)

    if not retrieved_chunks:
        print("No relevant chunks found.")
        # 関連チャンクが見つからなかった場合でもLLMにクエリだけ投げるか、
        # それとも「情報が見つかりませんでした」とだけ答えるか、研究目的に応じて調整
        # ここでは情報が見つからなかったことをコンテキストで伝える形式にする
        context = format_context([])  # 関連情報なしのコンテキストを生成
    else:
        # 取得したチャンクをコンテキスト形式に整形
        context = format_context(retrieved_chunks)
        print("Context formatted.")

    # LLMへのプロンプトを作成
    # システムプロンプトでLLMの振る舞いを指示
    # 参照情報に基づき回答するよう明確に指示するのがRAGのポイント
    system_prompt = (
        "あなたはユーザーの質問に対し、提供された【参照情報】のみを用いて回答するAIアシスタントです。\n"
        "【参照情報】に含まれていない内容については推測で答えたり、知らないと正直に答えてください。\n"
        "回答は丁寧で分かりやすい言葉遣いを心がけてください。\n"
    )

    # ユーザーの質問と参照情報を組み合わせたプロンプト
    user_prompt = f"質問：{query}\n\n{context}"

    # Ollamaで回答を生成
    print(f"Generating response using '{LLM_MODEL}'...")
    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        llm_response = response["message"]["content"]
        print("Response generated successfully.")
        return llm_response

    except ollama.ResponseError as e:
        print(f"Error generating response with Ollama: {e}")
        return f"回答生成中にOllamaでエラーが発生しました: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during response generation: {e}")
        return f"回答生成中にエラーが発生しました: {e}"


# --- スクリプトとして実行された場合の処理 ---
if __name__ == "__main__":
    # 処理済みのRAGデータを読み込む
    processed_rag_data = load_processed_data(PROCESSED_DATA_FILE)

    if processed_rag_data:
        # テスト用のクエリ
        test_query = "筑波大学の生成AIに関するガイドラインの要点は何ですか？"
        # test_query = "この文書には何について書かれていますか？" # 別のクエリを試すことも可能

        print(f'\n--- Processing Query: "{test_query}" ---')

        # RAG検索と回答生成を実行
        final_response = get_rag_response(test_query, processed_rag_data)

        # 結果を表示
        print("\n--- Generated Response (RAG) ---")
        if final_response:
            print(final_response)
        else:
            print("回答を生成できませんでした。")
        print("\n" + "=" * 30 + "\n")
