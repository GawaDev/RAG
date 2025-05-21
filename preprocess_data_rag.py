# preprocess_data_rag.py

import json
import os
from typing import Any, Dict, List

import numpy as np
import ollama

# --- 設定 ---
# 事前入力データが配置されているディレクトリを指定してください
DATA_DIR = "data/raw"
# RAGで使用するEmbeddingモデル名
# Ollamaで pull済みのモデルを指定 (例: nomic-embed-text, mistralなど)
# mistralはEmbeddingも可能ですが、nomic-embed-textのような専用モデルの方が適していることが多いです。
# ここでは例として nomic-embed-text を使用します。もし持っていなければ 'ollama pull nomic-embed-text' してください。
EMBEDDING_MODEL = "nomic-embed-text"
# テキストをチャンクに分割するおおよその最大文字数
CHUNK_SIZE = 500
# チャンク間に持たせるオーバーラップ文字数
CHUNK_OVERLAP = 100

# --- ヘルパー関数 ---


def read_file_content(filepath: str) -> str:
    """
    指定されたファイルのコンテンツを読み込む。
    現在のところ、シンプルなテキストファイルとJSONファイルを扱う。
    """
    try:
        # エンコーディングを自動判定または一般的なものを試す
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # JSONの場合は整形して読みやすくすることも考えられるが、ここではシンプルに文字列として扱う
        # JSONとしてパースして特定のフィールドだけ使う場合は別途ロジックが必要
        return content
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="shift_jis") as f:
                content = f.read()
            return content
        except Exception as e:
            print(
                f"Warning: Could not read file {filepath} with utf-8 or shift_jis: {e}"
            )
            return ""
    except Exception as e:
        print(f"Warning: Could not read file {filepath}: {e}")
        return ""


def simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    シンプルなチャンク分割関数。
    テキストを一定のサイズで分割し、オーバーラップを持たせる。
    句読点などを考慮しない非常に基本的な分割。
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # チャンクの末尾がテキスト全体の長さを超えないように調整
        chunk = text[start : min(end, len(text))]
        chunks.append(chunk)

        # 次のチャンクの開始位置を計算
        # テキストの終端に達した場合はループを終了
        if end >= len(text):
            break

        # オーバーラップを考慮して開始位置を進める
        start += chunk_size - chunk_overlap

        # オーバーラップがチャンクサイズより大きいなど、不正な設定の対応
        if chunk_overlap >= chunk_size and start < len(text):
            # オーバーラップが大きすぎる場合は警告またはエラー処理が必要
            # ここでは単純にオーバーラップなしで次へ進むなどのフォールバックも考えられる
            pass  # より堅牢な実装ではここに対策が必要

    return chunks


# --- メイン処理 ---


def preprocess_for_rag(
    data_dir: str = DATA_DIR, embedding_model: str = EMBEDDING_MODEL
) -> List[Dict[str, Any]]:
    """
    指定ディレクトリのファイルを読み込み、チャンク分割とベクトル埋め込みを行う。
    """
    print(f"--- Starting RAG Preprocessing for directory: {data_dir} ---")

    all_chunks_with_vectors: List[Dict[str, Any]] = []
    ollama_client = None

    try:
        # Ollamaクライアント初期化
        ollama_client = ollama.Client()
        # 指定したEmbeddingモデルが利用可能か確認 (pingのようなもの)
        # 厳密な確認はAPIエラーハンドリングで行う
        print(f"Using Ollama model for embeddings: {embedding_model}")

    except Exception as e:
        print(f"Error initializing Ollama client: {e}")
        print("Cannot proceed with embedding generation.")
        return []  # クライアント初期化失敗時は空リストを返す

    # データディレクトリが存在しない場合は作成または警告
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return []

    # ディレクトリ内のファイルを走査
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        # ディレクトリではなくファイルであることを確認
        if os.path.isfile(filepath):
            print(f"Processing file: {filename}")

            # ファイルコンテンツ読み込み
            content = read_file_content(filepath)
            if not content:
                print(f"Skipping empty or unreadable file: {filename}")
                continue

            # シンプルなチャンク分割
            chunks = simple_chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f" - Split into {len(chunks)} chunks.")

            # 各チャンクのベクトル埋め込みを生成
            chunk_vectors = []
            try:
                # Ollamaのembeddingsエンドポイントを呼び出す
                # 複数のテキストを一度に処理できるAPIがある場合そちらが効率的
                # ここでは簡単のためチャンクごとにループ
                for i, chunk in enumerate(chunks):
                    # プログレス表示
                    if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                        print(
                            f"   Generating embedding for chunk {i + 1}/{len(chunks)}..."
                        )

                    response = ollama_client.embeddings(
                        model=embedding_model,
                        prompt=chunk,  # 埋め込み生成対象のテキスト
                    )
                    # レスポンスからベクトル情報を取り出す
                    vector = response["embedding"]
                    chunk_vectors.append(vector)

                    # チャンクとベクトルを一緒に保存
                    all_chunks_with_vectors.append(
                        {
                            "text": chunk,
                            "vector": vector,
                            "source": filename,  # 元ファイル名
                            "chunk_id": i,  # ファイル内のチャンク番号
                        }
                    )

            except ollama.ResponseError as e:
                print(f"Error generating embeddings for {filename}: {e}")
                print("Skipping embeddings for this file.")
                # このファイルのチャンクはベクトルなしでスキップされる
                continue  # 次のファイルに進む
            except Exception as e:
                print(
                    f"An unexpected error occurred during embedding generation for {filename}: {e}"
                )
                print("Skipping embeddings for this file.")
                continue  # 次のファイルに進む

    print(
        f"--- RAG Preprocessing Complete. Total chunks processed: {len(all_chunks_with_vectors)} ---"
    )

    # 結果の確認（最初の数件を表示）
    print("\n--- Sample Processed Chunks ---")
    for i, item in enumerate(all_chunks_with_vectors[:5]):  # 最初の5件
        print(f"Chunk {i + 1} (from {item['source']}, id {item['chunk_id']}):")
        print(f'  Text: "{item["text"][:100]}..."')  # テキストの先頭100文字
        print(f"  Vector shape: {np.array(item['vector']).shape}")  # ベクトルの形状
        print("-" * 20)

    return all_chunks_with_vectors


# --- スクリプトとして実行された場合の処理 ---
if __name__ == "__main__":
    # 事前入力データディレクトリが存在しない場合は作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")
        print(
            f"Please place your .txt or .json files inside the '{DATA_DIR}' directory and run the script again."
        )
    else:
        # ディレクトリ内のファイルが空でないか確認
        if not any(
            os.path.isfile(os.path.join(DATA_DIR, name))
            for name in os.listdir(DATA_DIR)
        ):
            print(f"Data directory '{DATA_DIR}' is empty.")
            print(
                f"Please place your .txt or .json files inside the '{DATA_DIR}' directory."
            )
        else:
            # 前処理を実行してデータを取得
            processed_rag_data = preprocess_for_rag(DATA_DIR, EMBEDDING_MODEL)

            # --- 追加ここから ---
            # 処理結果を一時ファイルに保存
            PROCESSED_DATA_FILE = "processed_rag_data.json"
            if processed_rag_data:  # データが空でない場合のみ保存
                try:
                    # NumPyのndarrayはJSONEncoderで直接扱えないためリストに変換
                    serializable_data = [
                        {
                            "text": item["text"],
                            "vector": np.array(
                                item["vector"]
                            ).tolist(),  # NumPy配列をリストに変換
                            "source": item["source"],
                            "chunk_id": item["chunk_id"],
                        }
                        for item in processed_rag_data
                    ]

                    with open(PROCESSED_DATA_FILE, "w", encoding="utf-8") as f:
                        json.dump(serializable_data, f, indent=2)
                    print(f"\nProcessed data saved to {PROCESSED_DATA_FILE}")
                except Exception as e:
                    print(f"Error saving processed data to file: {e}")
            else:
                print("\nNo data processed, skipping save.")
            # --- 追加ここまで ---

            print("\nProcessed data is ready for RAG search.")
