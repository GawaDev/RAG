# app.py

import json
import os
import sqlite3
import time  # 実行時間計測用
from typing import List

import numpy as np  # RAGデータ保存時のリスト変換に使用
import streamlit as st

# --- プロジェクト内のモジュールから関数をインポート ---
# 各ファイルに関数が定義されている前提です
# 必要に応じて、インポートする関数名やファイル名を調整してください

# preprocess_data_rag.py に以下の関数と定数があることを想定
try:
    from preprocess_data_rag import DATA_DIR as RAG_DATA_DIR
    from preprocess_data_rag import EMBEDDING_MODEL as RAG_EMBEDDING_MODEL
    from preprocess_data_rag import preprocess_for_rag
except ImportError:
    st.error(
        "Error: preprocess_data_rag.py が見つからないか、必要な関数/定数が定義されていません。"
    )
    st.stop()  # スクリプトを停止

# preprocess_graph.py に以下の関数と定数があることを想定
try:
    from neo4j import GraphDatabase  # Neo4j接続用

    from preprocess_graph import DATA_DIR as GRAPH_DATA_DIR
    from preprocess_graph import ERE_MODEL as GRAPH_ERE_MODEL
    from preprocess_graph import preprocess_for_graph
except ImportError:
    st.error(
        "Error: preprocess_graph.py が見つからないか、必要な関数/定数が定義されていません。Neo4jライブラリも確認してください。"
    )
    st.stop()  # スクリプトを停止


# rag_search_generate.py または rag_search.py に以下の関数があることを想定
# 関数名を get_rag_context に修正している前提です
try:
    # rag_search_generate.py 内の関数名を変更している可能性があります
    # find_similar_chunks と format_context 関数を使用
    from rag_search_generate import find_similar_chunks
    from rag_search_generate import format_context as format_rag_context
except ImportError:
    st.error(
        "Error: rag_search_generate.py が見つからないか、必要な関数が定義されていません。"
    )
    st.stop()  # スクリプトを停止

# graph_search.py に以下の関数があることを想定
# 関数名を get_graph_context に修正している前提です
try:
    # graph_search.py 内の関数を使用
    from graph_search import (
        format_graph_results_for_llm,
        identify_query_entities,
        perform_graph_search,
    )
except ImportError:
    st.error(
        "Error: graph_search.py が見つからないか、必要な関数が定義されていません。"
    )
    st.stop()  # スクリプトを停止

# llm_generate.py に generate_response 関数があることを想定
try:
    import ollama  # Ollamaクライアント用

    from llm_generate import generate_response
except ImportError:
    st.error(
        "Error: llm_generate.py が見つからないか、必要な関数が定義されていません。Ollamaライブラリも確認してください。"
    )
    st.stop()  # スクリプトを停止


# --- 設定 ---
# RAG処理済みデータを保存/ロードするファイル (データ解析結果をアプリ間で共有するため)
PROCESSED_RAG_DATA_FILE = "processed_rag_data_cache.json"  # キャッシュファイル名を変更
# 実行結果を保存するデータベースファイル
RESULTS_DB_FILE = "research_results.db"
# Neo4j接続情報 (環境変数から取得)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # 環境変数から取得

# --- Streamlit アプリケーションの定義 ---

st.set_page_config(layout="wide", page_title="RAG vs GraphRAG 研究システム")

# --- 状態管理 ---
# Streamlit のセッション状態を使って、アプリの状態を維持
if "processed_rag_data" not in st.session_state:
    # ファイルからキャッシュされたデータをロード、なければNone
    try:
        if os.path.exists(PROCESSED_RAG_DATA_FILE):
            with open(PROCESSED_RAG_DATA_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                # ベクトルはリストとして保存されているので、numpy配列に戻す必要はないが、
                # 念のためリスト形式であることを確認
                if (
                    cached_data
                    and isinstance(cached_data, list)
                    and "vector" in cached_data[0]
                    and isinstance(cached_data[0]["vector"], list)
                ):
                    st.session_state["processed_rag_data"] = cached_data
                    print(
                        f"Cached RAG data loaded from {PROCESSED_RAG_DATA_FILE}"
                    )  # ターミナルにログ
                else:
                    st.session_state["processed_rag_data"] = None
                    print(
                        f"Cached RAG data file {PROCESSED_RAG_DATA_FILE} is empty or invalid format."
                    )  # ターミナルにログ
        else:
            st.session_state["processed_rag_data"] = None
            print(
                f"No RAG data cache file found at {PROCESSED_RAG_DATA_FILE}"
            )  # ターミナルにログ

    except Exception as e:
        st.session_state["processed_rag_data"] = None
        print(f"Error loading cached RAG data: {e}")  # ターミナルにログ


if "graph_processed_status" not in st.session_state:
    # GraphRAGの処理済み状態はNeo4jのデータ存在で判断することも可能だが、
    # ここではシンプルにセッション中に解析ボタンを押したかを記録
    st.session_state["graph_processed_status"] = False  # アプリ起動時は未処理状態とする

if "ollama_client" not in st.session_state:
    st.session_state["ollama_client"] = None  # Ollamaクライアントインスタンス

if "ollama_models" not in st.session_state:
    st.session_state["ollama_models"] = []  # 利用可能なOllamaモデル名リスト

if "view_history" not in st.session_state:
    st.session_state["view_history"] = False  # 履歴表示状態フラグ


# --- データベース初期化 ---
def init_db():
    """結果保存用データベースとテーブルを作成/確認する"""
    try:
        conn = sqlite3.connect(RESULTS_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                method TEXT, -- 'RAG' or 'GraphRAG'
                model TEXT,
                rag_context_summary TEXT, -- RAGコンテキストの概要（例: チャンク数）
                graph_context_summary TEXT, -- Graphコンテキストの概要（例: ノード数、エッジ数）
                -- context_content TEXT, -- RAG/GraphRAGコンテキストの全内容 (保存するかは要検討、容量注意)
                response TEXT,
                processing_time_ms REAL, -- 処理時間（ミリ秒）
                source_data_dir TEXT -- どのデータディレクトリを使ったか
            )
        """)
        conn.commit()
        conn.close()
        # print(f"Database {RESULTS_DB_FILE} initialized.") # ターミナルにログ
    except Exception as e:
        st.error(f"データベースの初期化に失敗しました: {e}")
        # print(f"Database initialization error: {e}") # ターミナルにログ


# --- 各種サービスのクライアント初期化 ---
def init_clients():
    """Ollamaクライアントを初期化し、セッション状態に保存する"""
    if st.session_state["ollama_client"] is None:
        try:
            st.session_state["ollama_client"] = ollama.Client()
            st.sidebar.success("Ollamaクライアントに接続しました。")
            # 利用可能なOllamaモデルを取得
            models_list = st.session_state["ollama_client"].list()
            st.session_state["ollama_models"] = [
                m["name"] for m in models_list["models"]
            ]
            # st.sidebar.write(f"利用可能なOllamaモデル: {', '.join(st.session_state['ollama_models'])}") # サイドバーに表示
            # print(f"Available Ollama models: {st.session_state['ollama_models']}") # ターミナルにログ

        except Exception as e:
            st.sidebar.error(
                f"Ollamaクライアントの接続に失敗しました。Ollamaが起動しているか確認してください。\nエラー: {e}"
            )
            st.session_state["ollama_client"] = None
            st.session_state["ollama_models"] = []
            # print(f"Ollama connection error: {e}") # ターミナルにログ


# --- データ解析処理 ---
def perform_data_parsing():
    """「データ解析実行」ボタン押下時の処理"""
    st.subheader("データ解析状況")
    status_area = st.empty()  # リアルタイムで状況を表示するためのプレースホルダー
    status_area.info("データ解析を開始します...")

    ollama_client = st.session_state["ollama_client"]
    if ollama_client is None:
        status_area.error(
            "Ollamaクライアントが利用できません。接続を確認してください。"
        )
        return

    # --- RAG向け前処理 ---
    status_area.info(f"RAG向け前処理 ({RAG_DATA_DIR}) を実行中...")
    try:
        # preprocess_data_rag.py の preprocess_for_rag 関数を呼び出す
        rag_data = preprocess_for_rag(RAG_DATA_DIR, RAG_EMBEDDING_MODEL)
        st.session_state["processed_rag_data"] = (
            rag_data  # 処理結果をセッション状態に保存
        )

        # 処理済みRAGデータをファイルにキャッシュ保存
        if rag_data:
            try:
                serializable_data = [
                    {
                        "text": item["text"],
                        "vector": np.array(
                            item["vector"]
                        ).tolist(),  # NumPy配列をリストに変換
                        "source": item["source"],
                        "chunk_id": item["chunk_id"],
                    }
                    for item in rag_data
                ]
                with open(PROCESSED_RAG_DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(serializable_data, f, indent=2)
                # print(f"RAG processed data cached to {PROCESSED_RAG_DATA_FILE}") # ターミナルにログ
            except Exception as e:
                print(f"Error saving RAG processed data cache: {e}")  # ターミナルにログ

            status_area.success(
                f"RAG向け前処理完了。{len(rag_data)} チャンクを処理しました。"
            )
        else:
            st.session_state["processed_rag_data"] = None
            status_area.warning(
                "RAG向け前処理完了。処理対象ファイルが見つからないかエラーが発生しました。"
            )

    except Exception as e:
        status_area.error(f"RAG向け前処理中にエラーが発生しました: {e}")
        st.session_state["processed_rag_data"] = None
        # print(f"RAG preprocessing error: {e}") # ターミナルにログ

    # --- GraphRAG向け前処理 ---
    status_area.info(f"GraphRAG向け前処理 ({GRAPH_DATA_DIR}) を実行中...")
    # Neo4j接続は preprocess_for_graph 関数内で確立・切断される前提
    try:
        # preprocess_graph.py の preprocess_for_graph 関数を呼び出す
        # 返り値で成功/失敗を受け取る想定
        success = preprocess_for_graph(
            GRAPH_DATA_DIR, GRAPH_ERE_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        )
        st.session_state["graph_processed_status"] = (
            success  # 処理結果（成功/失敗）をセッションに保存
        )

        if success:
            status_area.success(
                "GraphRAG向け前処理完了。Neo4jにデータが登録されたはずです。"
            )
            # print("Graph preprocessing successful.") # ターミナルにログ
        else:
            status_area.warning(
                "GraphRAG向け前処理中に問題が発生した可能性があります。ターミナル出力を確認してください。"
            )
            # print("Graph preprocessing might have issues. Check terminal output.") # ターミナルにログ

    except Exception as e:
        status_area.error(f"GraphRAG向け前処理中にエラーが発生しました: {e}")
        st.session_state["graph_processed_status"] = False
        # print(f"Graph preprocessing error: {e}") # ターミナルにログ

    status_area.info("全てのデータ解析が完了しました。質問の実行に進めます。")


# --- 実行処理 ---
def perform_execution(query: str, methods: List[str], models: List[str]):
    """「実行」ボタン押下時の処理"""
    if not query:
        st.warning("質問文を入力してください。")
        return

    if not methods:
        st.warning("実行する手法（RAGまたはGraphRAG）を選択してください。")
        return

    if not models:
        st.warning("利用するOllamaモデルを選択してください。")
        return

    # 必要なデータやサービスが利用可能か確認
    if "RAG" in methods and (
        st.session_state["processed_rag_data"] is None
        or len(st.session_state["processed_rag_data"]) == 0
    ):
        st.warning(
            "RAG実行にはRAG向けデータ解析が必要です。データ解析を実行し、データが生成されたか確認してください。"
        )
        return

    if "GraphRAG" in methods:
        if not st.session_state["graph_processed_status"]:
            st.warning(
                "GraphRAG実行にはGraphRAG向けデータ解析が必要です。サイドバーの「データ解析実行」ボタンを押してください。"
            )
            return
        # Neo4j接続も確認が必要
        try:
            driver_check = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            driver_check.verify_connectivity()
            driver_check.close()  # 確認後切断
        except Exception as e:
            st.warning(
                f"GraphRAG実行前にNeo4jが起動しているか、接続情報が正しいか確認してください: {e}"
            )
            return

    ollama_client = st.session_state["ollama_client"]
    if ollama_client is None:
        st.error("Ollamaクライアントが利用できません。接続を確認してください。")
        return

    st.subheader("実行結果")
    # 実行中のステータスを表示するためのプレースホルダー
    execution_status_area = st.empty()
    execution_status_area.info("実行中...")

    all_results = []  # 今回の実行で得られた全ての結果を保持するリスト

    # Neo4jドライバーはGraphRAGが選択されている場合のみ作成し、実行後に閉じる
    neo4j_driver = None

    # 手法 x モデル の組み合わせでループして実行
    # 手法 x モデル の組み合わせでループして実行
    total_executions = len(methods) * len(models)
    completed_count = 0

    try:
        # GraphRAGが選択されている場合のみNeo4jドライバーを作成
        neo4j_driver = None  # 関数スコープ内で初期化
        if "GraphRAG" in methods:
            # Neo4j接続はここで確立し、関数終了時に閉じる
            try:
                neo4j_driver = GraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                neo4j_driver.verify_connectivity()  # 接続確認
                # print("Neo4j driver connected for execution.") # ターミナルにログ
            except Exception as e:
                # GraphRAGが選択されているがNeo4j接続できない場合のエラー処理
                # driver = None のままにし、以下のループ内でスキップ処理を行う
                st.warning(
                    f"GraphRAGが選択されましたが、Neo4jに接続できませんでした。確認してください: {e}"
                )
                # print(f"Neo4j connection error for GraphRAG execution: {e}") # ターミナルにログ

        for method in methods:
            for model_name in models:
                start_time = time.time()
                completed_count += 1
                # 実行中のステータス表示を更新
                current_status_text = f"実行中: 手法='{method}', モデル='{model_name}' ({completed_count}/{total_executions})..."
                execution_status_area.info(current_status_text)

                context_content = ""  # LLMに渡すコンテキストのテキスト内容
                context_summary = ""  # UI表示/DB保存用のコンテキスト概要
                response_text = (
                    "--- 回答生成スキップ ---"  # デフォルトスキップメッセージ
                )
                execution_status = "Skipped"  # 実行ステータス

                try:
                    # --- 各手法の実行可能性チェックとコンテキスト取得 ---
                    if method == "RAG":
                        if (
                            st.session_state["processed_rag_data"] is not None
                            and len(st.session_state["processed_rag_data"]) > 0
                        ):
                            # RAGデータが準備できていれば実行
                            print(
                                f"Executing RAG for {model_name}..."
                            )  # ターミナルにログ
                            # rag_search_generate.py の関数を呼び出し RAG コンテキスト取得
                            query_vector_res = ollama_client.embeddings(
                                model=RAG_EMBEDDING_MODEL, prompt=query
                            )
                            query_vector = query_vector_res["embedding"]
                            retrieved_chunks = find_similar_chunks(
                                query_vector, st.session_state["processed_rag_data"]
                            )
                            context_content = format_rag_context(retrieved_chunks)
                            context_summary = f"{len(retrieved_chunks)} チャンク取得"
                            if not retrieved_chunks:
                                context_summary += " (関連なし)"

                            # LLMで回答生成
                            response_text = generate_response(
                                query, context_content, model_name, ollama_client
                            )
                            execution_status = (
                                "Success" if response_text is not None else "Failed_LLM"
                            )

                        else:
                            # RAGデータが準備できていない場合はスキップ
                            context_content = "【参照情報】\nRAGデータがありません。データ解析を確認してください。"
                            context_summary = "RAGデータなし"
                            response_text = "RAGデータが準備されていないため、回答を生成できませんでした。"
                            execution_status = "Skipped_Data"
                            # print(f"Skipping RAG for {model_name}: RAG data not available.") # ターミナルにログ

                    elif method == "GraphRAG":
                        # GraphRAGが選択されている場合、処理済み状態とNeo4j接続をチェック
                        if not st.session_state["graph_processed_status"]:
                            # Graph前処理が完了していない場合スキップ
                            context_content = "【Graph参照情報】\nGraphRAG向けデータ解析が完了していません。"
                            context_summary = "Graph未解析"
                            response_text = "GraphRAG向けデータ解析が完了していないため、回答を生成できませんでした。"
                            execution_status = "Skipped_Preprocessing"
                            # print(f"Skipping GraphRAG for {model_name}: Graph preprocessing not done.") # ターミナルにログ

                        elif neo4j_driver is None:
                            # Neo4j接続が確立できなかった場合スキップ (try/exceptで捕捉済みのはずだが念のため)
                            context_content = (
                                "【Graph参照情報】\nNeo4jに接続できません。"
                            )
                            context_summary = "Neo4j接続エラー"
                            response_text = "Neo4jに接続できないため、GraphRAGを実行できませんでした。"
                            execution_status = "Skipped_Neo4j"
                            # print(f"Skipping GraphRAG for {model_name}: Neo4j driver not available.") # ターミナルにログ

                        else:
                            # Graphデータが準備できており、Neo4jに接続できれば実行
                            print(
                                f"Executing GraphRAG for {model_name}..."
                            )  # ターミナルにログ
                            # graph_search.py の関数を呼び出し GraphRAG コンテキスト取得
                            query_entities = identify_query_entities(
                                query, ollama_client, GRAPH_ERE_MODEL
                            )  # クエリ分析 (Ollama利用)
                            if not query_entities:
                                # クエリからエンティティが特定できない場合
                                context_content = "【Graph参照情報】\nクエリからGraph検索の起点となるエンティティを特定できませんでした。"
                                context_summary = "エンティティ特定失敗"
                                response_text = "クエリから関連エンティティを特定できないため、GraphRAGを実行できませんでした。"
                                execution_status = "Failed_EntityID"
                                # print(f"GraphRAG failed for {model_name}: Entity identification failed.") # ターミナルにログ
                            else:
                                # Graph検索を実行
                                graph_result = perform_graph_search(
                                    query_entities, neo4j_driver
                                )
                                # コンテキスト整形
                                context_content = format_graph_results_for_llm(
                                    graph_result
                                )
                                # Graphコンテキストの概要を作成（例: 取得ノード数、関係性数）
                                num_nodes = sum(
                                    1 for item in graph_result if "labels" in item
                                )
                                num_rels = sum(
                                    1 for item in graph_result if "type" in item
                                )  # Neo4j 5.x driver result might need adjustment
                                context_summary = (
                                    f"ノード: {num_nodes}, 関係性: {num_rels} 取得"
                                )

                                if num_nodes == 0 and num_rels == 0:
                                    # Graph検索で結果が見つからなかった場合
                                    context_summary += " (関連なし)"
                                    response_text = "GraphRAG検索では関連情報が見つかりませんでした。"
                                    execution_status = "Success_NoContext"  # 実行は成功したがコンテキストは得られず

                                else:
                                    # LLMで回答生成 (Graphコンテキストとクエリを使用)
                                    response_text = generate_response(
                                        query,
                                        context_content,
                                        model_name,
                                        ollama_client,
                                    )
                                    execution_status = (
                                        "Success"
                                        if response_text is not None
                                        else "Failed_LLM"
                                    )

                except Exception as e:
                    # 個別の手法実行中に予期せぬエラー
                    response_text = (
                        f"実行中にエラーが発生しました ({method}/{model_name}): {e}"
                    )
                    context_summary = "実行エラー"
                    execution_status = "Error"
                    print(
                        f"Execution error for {method}/{model_name}: {e}"
                    )  # ターミナルにログ

                # --- 結果をデータベースに保存 ---
                # スキップされた実行もDBに記録する
                try:
                    conn = sqlite3.connect(RESULTS_DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO results (query, method, model, rag_context_summary, graph_context_summary, response, processing_time_ms, source_data_dir)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            query,
                            method,
                            model_name,
                            context_summary
                            if method == "RAG"
                            else "",  # RAGならRAG概要
                            context_summary
                            if method == "GraphRAG"
                            else "",  # GraphRAGならGraph概要
                            # context_content, # コンテキスト内容自体をDBに入れるかはお好みで（容量注意）
                            response_text,  # 回答またはエラー/スキップメッセージ
                            processing_time_ms,
                            RAG_DATA_DIR,  # 今回使ったデータディレクトリ（共通として扱う）
                        ),
                    )
                    conn.commit()
                    conn.close()
                    # print(f"Result saved to DB for {method}/{model_name}") # ターミナルにログ
                except Exception as e:
                    st.error(
                        f"データベースへの保存に失敗しました ({method}/{model_name}): {e}"
                    )
                    print(f"Database save error for {method}/{model_name}: {e}")

                # 今回の実行結果を一時リストに追加（画面表示用）
                all_results.append(
                    {
                        "method": method,
                        "model": model_name,
                        "context_summary": context_summary,
                        "response": response_text,
                        "time_ms": processing_time_ms,
                        "status": execution_status,  # 実行ステータスも追加
                    }
                )

        # 全ての実行が完了
        execution_status_area.empty()  # 進行状況表示をクリア

        st.subheader("今回の実行結果まとめ")
        # 結果を整形して表示
        for i, result in enumerate(all_results):
            # ステータスによって色を変えるなどUIを工夫可能
            status_color = (
                "green"
                if result["status"] == "Success"
                else (
                    "orange"
                    if result["status"].startswith("Skipped")
                    or result["status"].endswith("NoContext")
                    else "red"
                )
            )

            st.markdown(
                f"<h4 style='color:{status_color};'>実行 {i + 1}. 手法: {result['method']}, モデル: {result['model']}</h4>",
                unsafe_allow_html=True,
            )
            st.write(
                f"**ステータス:** {result['status']}, **時間:** {result['time_ms']:.2f} ms"
            )
            st.write(f"**コンテキスト概要:** {result['context_summary']}")
            st.write("**回答:**")
            # 回答/メッセージの内容に応じて表示方法を調整
            if result["status"] == "Success" or result["status"] == "Success_NoContext":
                st.info(result["response"])  # 成功系の場合はinfoボックス
            elif result["status"].startswith("Skipped"):
                st.warning(result["response"])  # スキップされた場合はwarningボックス
            else:  # エラーの場合など
                st.error(result["response"])  # エラーの場合はerrorボックス

            st.markdown("---")  # 区切り線

    except Exception as e:
        # 実行ループ全体での予期せぬエラー
        st.error(f"実行処理中に予期せぬエラーが発生しました: {e}")
        print(f"Overall execution error: {e}")
    finally:
        # GraphRAG実行で使用したドライバーを閉じる
        if neo4j_driver:
            neo4j_driver.close()
            # print("Neo4j driver closed after execution.") # ターミナルにログ


# --- 履歴表示 ---
def view_history():
    """結果履歴を表示する"""
    st.subheader("実行履歴")
    try:
        conn = sqlite3.connect(RESULTS_DB_FILE)
        # row_factory を sqlite3.Row にするとカラム名でアクセスできるようになる
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM results ORDER BY timestamp DESC"
        )  # 新しい順に取得
        history_data = cursor.fetchall()
        conn.close()

        if history_data:
            # カラム名を動的に取得
            columns = history_data[0].keys() if history_data else []
            # pandas DataFrame に変換して表示するのがStreamlitでは最も簡単
            import pandas as pd

            df = pd.DataFrame(history_data, columns=columns)

            # オプション：フィルタリング機能などをここに追加可能
            # 例: 質問文でフィルタリング
            # search_query = st.text_input("質問文でフィルタ", "")
            # if search_query:
            #     df = df[df['query'].str.contains(search_query, case=False, na=False)]

            st.dataframe(df)  # 全データを表示 (フィルタリング後も含む)

        else:
            st.info("まだ実行履歴はありません。質問を実行すると履歴が保存されます。")

    except Exception as e:
        st.error(f"履歴の読み込みに失敗しました: {e}")
        # print(f"History view error: {e}") # ターミナルにログ


# --- メインアプリのエントリーポイント ---
def main():
    """Streamlitアプリのメイン関数"""
    init_db()  # データベース初期化
    init_clients()  # サービス クライアント初期化 (Ollamaのみ)

    # サイドバー
    with st.sidebar:
        st.header("設定")
        st.write(f"**データディレクトリ (RAG):** `{RAG_DATA_DIR}`")
        st.write(f"**データディレクトリ (Graph):** `{GRAPH_DATA_DIR}`")
        st.write(f"**RAG Embedding Model:** `{RAG_EMBEDDING_MODEL}`")
        st.write(f"**Graph ERE Model:** `{GRAPH_ERE_MODEL}`")
        st.write(f"**Neo4j URI:** `{NEO4J_URI}`")
        st.write("---")

        st.subheader("データ解析")
        help_text_parse = (
            f"指定ディレクトリ (`{RAG_DATA_DIR}`と`{GRAPH_DATA_DIR}`) からデータを読み込み、\n"
            "RAG用前処理（チャンク分割、Embedding生成）と\n"
            "Graph用前処理（エンティティ・関係性抽出、Neo4j登録）を実行します。\n"
            "RAGデータは一時ファイル (`{PROCESSED_RAG_DATA_FILE}`) にキャッシュされます。\n"
            "GraphデータはNeo4jに登録されます。"
        )
        if st.button("データ解析実行", help=help_text_parse):
            perform_data_parsing()

        st.write("---")

        st.subheader("利用可能Ollamaモデル")
        if st.session_state["ollama_models"]:
            st.write(", ".join(st.session_state["ollama_models"]))
        else:
            st.warning(
                "Ollamaモデルが取得できません。Ollamaが起動しているか確認してください。"
            )

        st.write("---")

        st.subheader("履歴表示")
        # ボタンで履歴表示/非表示を切り替える
        if st.button("実行履歴を表示"):
            st.session_state["view_history"] = True
        if st.button("質問入力に戻る"):
            st.session_state["view_history"] = False

        st.write("---")
        st.write("**現在のデータ/サービス状態:**")
        rag_status = (
            "ロード済み"
            if st.session_state["processed_rag_data"] is not None
            and len(st.session_state["processed_rag_data"]) > 0
            else "未ロード"
        )
        graph_status = (
            "処理済み" if st.session_state["graph_processed_status"] else "未処理"
        )
        ollama_status = (
            "接続済み" if st.session_state["ollama_client"] is not None else "未接続"
        )

        st.write(
            f"- RAGデータ: {rag_status} ({len(st.session_state['processed_rag_data']) if st.session_state['processed_rag_data'] is not None else 0} chunks)"
        )
        st.write(f"- Graphデータ: {graph_status}")
        st.write(f"- Ollama: {ollama_status}")
        # Neo4jの接続状態は実行時または解析時に確認

    # メインエリア
    if st.session_state.get("view_history", False):
        # 履歴表示フラグがTrueの場合、履歴表示関数を呼び出す
        view_history()
    else:
        # 履歴表示フラグがFalseの場合、質問・実行UIを表示
        st.title("RAG vs GraphRAG 研究システム")
        st.write("事前入力データに基づき、異なる手法とLLMモデルでの回答を比較します。")
        st.write("サイドバーの「データ解析実行」をまず行ってください。")

        st.subheader("質問と実行")
        query_text = st.text_area("質問文を入力してください:", height=100)

        col1, col2 = st.columns(2)
        with col1:
            selected_methods = st.multiselect(
                "実行する手法:",
                ["RAG", "GraphRAG"],
                default=["RAG", "GraphRAG"],
                help="比較したい情報検索手法を選択してください。",
            )
        with col2:
            # 利用可能モデルのリストを取得して表示
            available_models = st.session_state["ollama_models"]
            # デフォルトで最初のモデルを選択、リストが空の場合は空リスト
            default_models = available_models[:1] if available_models else []
            # もしmistralが含まれていればデフォルトにするなど、モデルリストに応じて調整可能
            if "mistral" in available_models:
                default_models = ["mistral"]
            elif available_models:
                default_models = [available_models[0]]

            selected_models = st.multiselect(
                "利用するOllamaモデル:",
                available_models,
                default=default_models,
                help="回答生成に使用するOllamaモデルを複数選択できます。",
            )

        # 実行ボタン
        help_text_execute = "選択した手法とモデルの組み合わせごとに回答を生成し、結果を履歴に保存します。"
        if st.button("実行", help=help_text_execute):
            # 実行前に、RAGデータまたはGraphデータが準備されているか再度簡易チェック
            if not selected_methods or not selected_models:
                st.warning("手法とモデルを選択してください。")
            elif "RAG" in selected_methods and (
                st.session_state["processed_rag_data"] is None
                or len(st.session_state["processed_rag_data"]) == 0
            ):
                st.warning(
                    "RAG実行にはデータ解析が必要です。サイドバーの「データ解析実行」ボタンを押してください。"
                )
            elif "GraphRAG" in selected_methods:
                # GraphRAG選択時はGraph処理済み状態とNeo4j接続可能性をチェック
                if not st.session_state["graph_processed_status"]:
                    st.warning(
                        "GraphRAG実行にはデータ解析が必要です。サイドバーの「データ解析実行」ボタンを押してください。"
                    )
                else:
                    # Neo4j接続はperform_execution内で確立しますが、ここでは警告を出すために事前に接続を試みる
                    try:
                        driver_check = GraphDatabase.driver(
                            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
                        )
                        driver_check.verify_connectivity()
                        driver_check.close()
                        # 全てのチェックがOKなら実行
                        perform_execution(query_text, selected_methods, selected_models)
                    except Exception as e:
                        st.warning(
                            f"GraphRAG実行前にNeo4jが起動しているか、接続情報が正しいか確認してください: {e}"
                        )

            else:  # RAGのみ選択の場合など、すべてのチェックがOKなら実行
                perform_execution(query_text, selected_methods, selected_models)


# --- スクリプト実行 ---
if __name__ == "__main__":
    main()
