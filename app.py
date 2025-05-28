import streamlit as st

st.set_page_config(layout="wide", page_title="RAG / GraphRAG 研究システム")

import json
import os
import sqlite3
import time
from datetime import datetime

import ollama
from neo4j import GraphDatabase

from graph_search import get_graph_context
from llm_generate import generate_response

# 各モジュールのインポート
from preprocess_data_rag import preprocess_for_rag
from preprocess_graph import preprocess_for_graph
from rag_search import (
    get_rag_context,
)

OLLAMA_HOST = st.secrets["ollama"]["host"]
NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USER = st.secrets["neo4j"]["user"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]

RAG_DATA_DIR = "./data/raw"
RESULTS_DB_FILE = "./research_results.db"
RAG_CACHE_FILE = "./processed_rag_data_cache.json"

# --- LLM モデル設定 ---
RAG_EMBEDDING_MODEL = "nomic-embed-text"  # RAGベクトル生成用モデル
GRAPH_ERE_MODEL = "llama3"  # Graphエンティティ・関係性抽出用モデル
LLM_MODELS = ["llama3", "llama4", "mistral", "gemma3", "qwen3", "phi4"]


# --- データベース初期化 ---
def init_db():
    conn = sqlite3.connect(RESULTS_DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            method TEXT NOT NULL,
            model TEXT NOT NULL,
            rag_context_summary TEXT,
            graph_context_summary TEXT,
            response TEXT,
            processing_time_ms REAL,
            source_data_dir TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# --- クライアント初期化 ---
@st.cache_resource
def init_clients():
    try:
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        test_response = ollama_client.chat(
            model="llama3",
            messages=[{"role": "user", "content": "ping"}],
            options={"temperature": 0.0},
        )
        st.success("Ollama client connected successfully.")
        return ollama_client
    except Exception as e:
        st.error(
            f"Ollama client connection failed: {e}. Please ensure Ollama is running at {OLLAMA_HOST}"
        )
        return None


# --- データ解析処理 ---
def perform_data_parsing():
    st.session_state["rag_processed_status_msg"] = "未実行"
    st.session_state["graph_processed_status_msg"] = "未実行"
    st.session_state["rag_processed_status"] = False
    st.session_state["graph_processed_status"] = False
    st.session_state["processed_rag_data"] = None

    if not os.path.exists(RAG_DATA_DIR):
        st.error(f"データディレクトリが見つかりません: {RAG_DATA_DIR}")
        return

    parsing_status_area = st.empty()
    parsing_status_area.info("データ解析を開始しています...")

    ollama_client = init_clients()
    if ollama_client is None:
        parsing_status_area.error(
            "Ollamaクライアントが利用できないため、データ解析を開始できません。"
        )
        return

    # RAGデータ前処理
    try:
        parsing_status_area.info("RAG向けデータ解析中 (チャンク化とEmbedding生成)...")
        processed_data = preprocess_for_rag(
            RAG_DATA_DIR, RAG_EMBEDDING_MODEL, ollama_client
        )
        if processed_data:
            st.session_state["processed_rag_data"] = processed_data
            st.session_state["rag_processed_status"] = True
            st.session_state["rag_processed_status_msg"] = (
                f"RAG向け前処理完了 ({len(processed_data)}チャンク)。"
            )
            parsing_status_area.success(st.session_state["rag_processed_status_msg"])
            # キャッシュに保存
            with open(RAG_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            st.info(f"RAGデータは '{RAG_CACHE_FILE}' にキャッシュされました。")
        else:
            st.session_state["rag_processed_status"] = False
            st.session_state["rag_processed_status_msg"] = (
                "RAG向け前処理に失敗しました。"
            )
            parsing_status_area.error(st.session_state["rag_processed_status_msg"])
    except Exception as e:
        st.session_state["rag_processed_status"] = False
        st.session_state["rag_processed_status_msg"] = f"RAG向け前処理中にエラー: {e}"
        parsing_status_area.error(st.session_state["rag_processed_status_msg"])

    # Graphデータ前処理
    try:
        parsing_status_area.info(
            "GraphRAG向けデータ解析中 (エンティティ・関係性抽出、Neo4j登録)..."
        )
        graph_processing_successful = preprocess_for_graph(
            RAG_DATA_DIR,
            GRAPH_ERE_MODEL,
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            ollama_client,
        )
        if graph_processing_successful:
            st.session_state["graph_processed_status"] = True
            st.session_state["graph_processed_status_msg"] = "GraphRAG向け前処理完了。"
            parsing_status_area.success(st.session_state["graph_processed_status_msg"])
        else:
            st.session_state["graph_processed_status"] = False
            st.session_state["graph_processed_status_msg"] = (
                "GraphRAG向け前処理に失敗しました。"
            )
            parsing_status_area.error(st.session_state["graph_processed_status_msg"])
    except Exception as e:
        st.session_state["graph_processed_status"] = False
        st.session_state["graph_processed_status_msg"] = (
            f"GraphRAG向け前処理中にエラー: {e}"
        )
        parsing_status_area.error(st.session_state["graph_processed_status_msg"])

    st.toast("データ解析プロセスが完了しました。")
    parsing_status_area.empty()
    st.info("データ解析プロセスが完了しました。最新のステータスを確認してください。")


# --- 実行処理 ---
def perform_execution(query, methods, models):
    if not query:
        st.warning("質問を入力してください。")
        return
    if not methods:
        st.warning("少なくとも1つの手法を選択してください。")
        return
    if not models:
        st.warning("少なくとも1つのモデルを選択してください。")
        return

    ollama_client = init_clients()
    if ollama_client is None:
        st.error("Ollamaクライアントが利用できないため、質問実行不可。")
        return

    all_results = []
    execution_status_area = st.empty()

    # GraphRAGが選択されている場合のみNeo4jドライバーを作成
    neo4j_driver = None
    if "GraphRAG" in methods:
        try:
            neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            neo4j_driver.verify_connectivity()
            print("Neo4j driver connected for execution.")
        except Exception as e:
            # GraphRAGが選択されているがNeo4j接続できない場合のエラー処理
            st.warning(f"GraphRAGが選択されたが、Neo4jに接続不可。: {e}")
            print(f"GraphRAGが選択されたが、Neo4jに接続不可。: {e}")
            neo4j_driver = (
                None  # 接続失敗時はNoneのままにし、以下のループ内でスキップ処理を行う
            )

    # 手法 x モデル の組み合わせでループして実行
    total_executions = len(methods) * len(models)
    completed_count = 0

    try:
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
                    # --- 各手法の実行チェックとコンテキスト取得 ---
                    if method == "RAG":
                        if (
                            st.session_state["processed_rag_data"] is not None
                            and len(st.session_state["processed_rag_data"]) > 0
                        ):
                            # RAGデータが準備できていれば実行
                            print(
                                f"Executing RAG for {model_name}..."
                            )  # ターミナルにログ
                            # rag_search.py の関数を呼び出し RAG コンテキスト取得
                            context_content = get_rag_context(
                                query,
                                st.session_state["processed_rag_data"],
                                ollama_client,
                            )

                            # context_contentに基づいて概要を生成 (例: チャンク数、内容の有無)
                            if (
                                "RAGデータがありません。" in context_content
                                or "クエリのベクトル化に失敗しました" in context_content
                            ):
                                context_summary = "RAGデータなし/エラー"
                                response_text = "RAGデータが準備されていないか、検索に失敗したため、回答を生成できませんでした。"
                                execution_status = "Failed_Context"
                            else:
                                context_summary = f"{len(context_content.split('チャンク:')) - 1} チャンク取得"  # 簡易的なチャンク数カウント
                                # LLMで回答生成
                                response_text = generate_response(
                                    query, context_content, model_name, ollama_client
                                )
                                execution_status = (
                                    "Success"
                                    if response_text is not None
                                    else "Failed_LLM"
                                )
                        else:
                            # RAGデータが準備できていない場合はスキップ
                            context_content = "【参照情報】\nRAGデータがありません。データ解析を確認してください。"
                            context_summary = "RAGデータなし"
                            response_text = "RAGデータが準備されていないため、回答を生成できませんでした。"
                            execution_status = "Skipped_Data"
                            print(
                                f"Skipping RAG for {model_name}: RAG data not available."
                            )  # ターミナルにログ

                    elif method == "GraphRAG":
                        # GraphRAGが選択されている場合、処理済み状態とNeo4j接続をチェック
                        if not st.session_state["graph_processed_status"]:
                            # Graph前処理が完了していない場合スキップ
                            context_content = "【Graph参照情報】\nGraphRAG向けデータ解析が完了していません。"
                            context_summary = "Graph未解析"
                            response_text = "GraphRAG向けデータ解析が完了していないため、回答を生成できませんでした。"
                            execution_status = "Skipped_Preprocessing"
                            print(
                                f"Skipping GraphRAG for {model_name}: Graph preprocessing not done."
                            )  # ターミナルにログ

                        elif neo4j_driver is None:
                            # Neo4j接続が確立できなかった場合スキップ
                            context_content = (
                                "【Graph参照情報】\nNeo4jに接続できません。"
                            )
                            context_summary = "Neo4j接続エラー"
                            response_text = "Neo4jに接続できないため、GraphRAGを実行できませんでした。"
                            execution_status = "Skipped_Neo4j"
                            print(
                                f"Skipping GraphRAG for {model_name}: Neo4j driver not available."
                            )  # ターミナルにログ

                        else:
                            # Graphデータが準備できており、Neo4jに接続できれば実行
                            print(
                                f"Executing GraphRAG for {model_name}..."
                            )  # ターミナルにログ
                            # graph_search.py の関数を呼び出し GraphRAG コンテキスト取得
                            context_content = get_graph_context(
                                query, neo4j_driver, ollama_client
                            )

                            # context_contentに基づいて概要を生成 (例: ノード・関係性数、内容の有無)
                            if (
                                "Graph検索の起点となるエンティティを特定できませんでした。"
                                in context_content
                            ):
                                context_summary = "エンティティ特定失敗"
                                response_text = "クエリから関連エンティティを特定できないため、GraphRAGを実行できませんでした。"
                                execution_status = "Failed_EntityID"
                            elif "関連情報が見つかりませんでした。" in context_content:
                                context_summary = "関連なし"
                                response_text = (
                                    "GraphRAG検索では関連情報が見つかりませんでした。"
                                )
                                execution_status = "Success_NoContext"  # 実行は成功したがコンテキストは得られず
                            elif (
                                "Neo4jに接続できません。" in context_content
                                or "ドライバーが利用できません" in context_content
                            ):
                                context_summary = "Neo4jエラー"
                                response_text = "Neo4j接続に問題があるため、GraphRAGを実行できませんでした。"
                                execution_status = "Failed_Neo4j"
                            else:
                                # 簡易的なコンテキスト概要の抽出 (例: "ノード: X, 関係性: Y 取得" のような文字列があれば)
                                # context_contentから解析できればより正確
                                context_summary_match = re.search(
                                    r"ノード:\s*(\d+).*関係性:\s*(\d+).*",
                                    context_content,
                                )
                                if context_summary_match:
                                    context_summary = f"ノード: {context_summary_match.group(1)}, 関係性: {context_summary_match.group(2)} 取得"
                                else:
                                    context_summary = "Graphコンテキスト取得済み"  # もっと良い方法があれば変更
                                # LLMで回答生成
                                response_text = generate_response(
                                    query, context_content, model_name, ollama_client
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

                end_time = time.time()
                processing_time_ms = (end_time - start_time) * 1000

                # --- 結果をデータベースに保存 ---
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
                            response_text,  # 回答またはエラー/スキップメッセージ
                            processing_time_ms,
                            RAG_DATA_DIR,
                        ),
                    )
                    conn.commit()
                    conn.close()
                    print(
                        f"Result saved to DB for {method}/{model_name}"
                    )  # ターミナルにログ
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
        for i, result in enumerate(all_results):
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
            if result["status"] == "Success" or result["status"] == "Success_NoContext":
                st.info(result["response"])
            elif result["status"].startswith("Skipped"):
                st.warning(result["response"])
            else:
                st.error(result["response"])
            st.markdown("---")

    except Exception as e:
        st.error(f"実行処理中に予期せぬエラーが発生しました: {e}")
        print(f"Overall execution error: {e}")
    finally:
        # GraphRAG実行で使用したドライバーを閉じる
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j driver closed after execution.")  # ターミナルにログ


# --- 履歴表示 ---
def view_history():
    st.subheader("実行履歴")
    try:
        conn = sqlite3.connect(RESULTS_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
        history_results = cursor.fetchall()
        conn.close()

        if history_results:
            # カラム名を動的に取得
            col_names = [description[0] for description in cursor.description]

            # Streamlitのexpandersを使って各結果を表示
            for i, row in enumerate(history_results):
                row_dict = dict(zip(col_names, row))

                # 実行日時をJSTに変換
                timestamp_utc = datetime.strptime(
                    row_dict["timestamp"], "%Y-%m-%d %H:%M:%S"
                )
                timestamp_jst = timestamp_utc.replace(tzinfo=None) + timedelta(hours=9)

                header_text = (
                    f"**クエリ:** {row_dict['query']} | **手法:** {row_dict['method']} | "
                    f"**モデル:** {row_dict['model']} | **日時:** {timestamp_jst.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                with st.expander(header_text):
                    st.write(f"**クエリ:** {row_dict['query']}")
                    st.write(f"**手法:** {row_dict['method']}")
                    st.write(f"**モデル:** {row_dict['model']}")
                    st.write(
                        f"**RAG コンテキスト概要:** {row_dict['rag_context_summary'] if row_dict['rag_context_summary'] else 'N/A'}"
                    )
                    st.write(
                        f"**Graph コンテキスト概要:** {row_dict['graph_context_summary'] if row_dict['graph_context_summary'] else 'N/A'}"
                    )
                    st.write(f"**回答:** {row_dict['response']}")
                    st.write(f"**処理時間:** {row_dict['processing_time_ms']:.2f} ms")
                    st.write(
                        f"**ソースデータディレクトリ:** {row_dict['source_data_dir']}"
                    )
                    st.write(
                        f"**記録日時 (JST):** {timestamp_jst.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                st.markdown("---")
        else:
            st.info("まだ実行履歴がありません。")
    except Exception as e:
        st.error(f"履歴の読み込み中にエラーが発生しました: {e}")


# --- メインアプリのエントリーポイント ---
def main():
    # セッションステートの初期化
    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "main"  # 'main' or 'history'
    if "rag_processed_status" not in st.session_state:
        st.session_state["rag_processed_status"] = False
    if "graph_processed_status" not in st.session_state:
        st.session_state["graph_processed_status"] = False
    if "rag_processed_status_msg" not in st.session_state:
        st.session_state["rag_processed_status_msg"] = "未実行"
    if "graph_processed_status_msg" not in st.session_state:
        st.session_state["graph_processed_status_msg"] = "未実行"
    if "processed_rag_data" not in st.session_state:
        st.session_state["processed_rag_data"] = None
        # キャッシュファイルからRAGデータをロードを試みる
        if os.path.exists(RAG_CACHE_FILE):
            try:
                with open(RAG_CACHE_FILE, "r", encoding="utf-8") as f:
                    st.session_state["processed_rag_data"] = json.load(f)
                    st.session_state["rag_processed_status"] = True
                    st.session_state["rag_processed_status_msg"] = (
                        f"RAGデータがキャッシュからロードされました ({len(st.session_state['processed_rag_data'])}チャンク)。"
                    )
                    st.toast("RAGデータがキャッシュからロードされました。")
            except Exception as e:
                st.warning(f"RAGキャッシュファイルのロードに失敗しました: {e}")
                st.session_state["rag_processed_status"] = False
                st.session_state["rag_processed_status_msg"] = (
                    "RAGキャッシュロード失敗。"
                )

    init_db()  # データベースの初期化

    st.title("RAG & GraphRAG 研究システム")

    # サイドバー
    with st.sidebar:
        st.header("設定と操作")

        # データ解析セクション
        st.subheader("1. データ解析")
        st.write(f"データディレクトリ: `{RAG_DATA_DIR}`")
        st.write(f"RAG Embedding モデル: `{RAG_EMBEDDING_MODEL}`")
        st.write(f"Graph ERE モデル: `{GRAPH_ERE_MODEL}`")

        st.info(f"RAG データ状態: {st.session_state['rag_processed_status_msg']}")
        st.info(f"Graph データ状態: {st.session_state['graph_processed_status_msg']}")

        if st.button(
            "データ解析実行",
            help="データディレクトリ内のファイルを解析し、RAGデータとGraphデータを準備します。Neo4j接続も必要です。",
        ):
            perform_data_parsing()

        st.markdown("---")

        # 履歴表示ボタン
        st.subheader("2. 実行履歴")
        if st.button(
            "実行履歴を表示", help="これまでの実行結果をデータベースから表示します。"
        ):
            st.session_state["current_view"] = "history"

        # メイン画面に戻るボタン
        if st.session_state["current_view"] == "history" and st.button(
            "メイン画面に戻る"
        ):
            st.session_state["current_view"] = "main"

        st.markdown("---")
        st.subheader("システム情報")
        st.write(f"Ollama Host: `{OLLAMA_HOST}`")
        st.write(f"Neo4j URI: `{NEO4J_URI}`")
        st.write(f"Neo4j User: `{NEO4J_USER}`")

    # メインコンテンツ
    if st.session_state["current_view"] == "main":
        st.subheader("質問入力と実行")

        query = st.text_area("質問を入力してください:", height=100, key="query_input")

        col1, col2 = st.columns(2)
        with col1:
            st.multiselect(
                "実行する手法を選択:",
                options=["RAG", "GraphRAG"],
                default=["RAG"],
                key="selected_methods",
            )
        with col2:
            st.multiselect(
                "使用するLLMモデルを選択:",
                options=LLM_MODELS,
                default=LLM_MODELS[0] if LLM_MODELS else [],
                key="selected_models",
            )

        if st.button("実行", type="primary"):
            with st.spinner("処理中..."):
                perform_execution(
                    query,
                    st.session_state["selected_methods"],
                    st.session_state["selected_models"],
                )
    elif st.session_state["current_view"] == "history":
        view_history()


# --- スクリプト実行 ---
if __name__ == "__main__":
    import re  # perform_execution 内の正規表現のためにインポート
    from datetime import timedelta  # view_history 内で必要

    # Ollamaクライアントの初期化をここでも実行（@st.cache_resourceのため一度だけ実行される）
    ollama_client_init = init_clients()
    if ollama_client_init is None:
        st.stop()  # クライアント接続失敗時はアプリを停止

    main()
