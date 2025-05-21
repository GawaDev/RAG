# graph_search.py

import os
from typing import Any, Dict, List

import ollama
from neo4j import GraphDatabase

# --- 設定 ---
# Neo4j接続情報 (preprocess_graph.py と同じ)
URI = "bolt://localhost:7687"  # または "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = os.getenv("NEO4J_PASSWORD", "HIROTAHIROTA")  # 環境変数から取得
# クエリ内のエンティティ特定に使用するOllamaモデル名 (ERE_MODELと同じで良い)
QUERY_ANALYSIS_MODEL = "mistral"  # または preprocess_graph.py で使用したモデル
# Graph検索でたどる関係性の最大ホップ数
MAX_HOPS = 2

# --- ヘルパー関数 ---


def identify_query_entities(
    query: str, ollama_client: ollama.Client, model_name: str
) -> List[str]:
    """
    Ollamaを使ってクエリテキストから主要なエンティティ名を特定する。
    特定されたエンティティ名は、Graph検索の起点として使用する。
    """
    # クエリから特定のタイプのエンティティを抽出するよう指示するプロンプト
    # Neo4jに格納されているエンティティタイプと一致させるのが望ましい
    prompt = f"""
以下の質問文から、Neo4jデータベースに格納されている可能性のある主要なエンティティ名（人名、組織名、場所、概念/キーワード、文書要素など）を特定してください。

回答は、特定したエンティティ名を1つにつき1行でリスト形式で出力してください。エンティティ名以外は含めないでください。

例:
質問: 筑波大学のガイドラインはいつ発行されましたか？
出力:
筑波大学
ガイドライン

質問: 生成AIに関する最新情報は何ですか？
出力:
生成AI

では、以下の質問文からエンティティ名を抽出してください：

{query}
"""
    print(f"Identifying query entities using '{model_name}'...")
    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled entity extraction AI. Extract entity names from user queries based on the provided instructions. Output only the list of entity names, one per line.",
                },
                {"role": "user", "content": prompt},
            ],
            # 温度を低めに設定し、安定した出力を目指す
            options={"temperature": 0.0},
        )
        # Ollamaの応答を改行で分割し、各行の空白を削除してリストにする
        entities = [
            line.strip()
            for line in response["message"]["content"].strip().split("\n")
            if line.strip()
        ]
        print(f"Identified entities in query: {entities}")
        return entities

    except ollama.ResponseError as e:
        print(f"Error identifying query entities with Ollama: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during query entity identification: {e}")
        return []


def perform_graph_search(
    query_entity_names: List[str],
    driver: GraphDatabase.driver,
    max_hops: int = MAX_HOPS,
) -> List[Dict[str, Any]]:
    """
    Neo4jに対してGraph検索を実行し、関連するノードと関係性を取得する。
    クエリで特定されたエンティティ名を起点に指定ホップ数までたどる。
    """
    if not query_entity_names or not driver:
        print("No query entities provided or Neo4j driver is not available.")
        return []

    print(
        f"Performing graph search starting from entities: {query_entity_names} (max hops: {max_hops})..."
    )

    # Neo4j検索クエリ (Cypher)
    # 指定された名前のノードを検索し、そこから指定ホップ数以内のノードと関係性を取得
    # RETURN 句でノードと関係性をまとめて返す
    # 重複を避けるため collect(DISTINCT ...) を使用
    query = f"""
    MATCH (start:Entity)
    WHERE start.name IN $entityNames
    MATCH (start)-[r*1..{max_hops}]-(related)
    UNWIND r AS rel // 関係性のリストを展開
    RETURN collect(DISTINCT start) AS startNodes, collect(DISTINCT related) AS relatedNodes, collect(DISTINCT rel) AS relationships
    """
    # 注: より高度なGraphRAGでは、関係性のタイプを考慮したり、重み付けをしたり、より複雑なグラフアルゴリズムを使用したりします。
    # ここでは関連ノード・エッジを単純に取得する基本的なクエリです。

    try:
        with driver.session() as session:
            result = session.run(
                query, entityNames=query_entity_names, maxHops=max_hops
            )
            # 結果は通常、リストのリストの形式で返されるため、処理が必要
            # ここでは最初のレコードを取得し、定義したエイリアス (startNodes, relatedNodes, relationships) でアクセス
            record = result.single()
            if record:
                # Neo4jの結果オブジェクトをPythonの扱える形式に変換
                # nodes() や relationship() メソッドを使って属性にアクセス可能
                start_nodes = [dict(node) for node in record["startNodes"]]
                related_nodes = [dict(node) for node in record["relatedNodes"]]
                relationships = [dict(rel) for rel in record["relationships"]]

                # 取得した全ノードと関係性をまとめる（重複削除しつつ）
                all_nodes = {}
                for node in start_nodes + related_nodes:
                    node_id = node.get(
                        "element_id"
                    )  # Neo4j 5.x 以降は element_id 推奨 id() は非推奨
                    if node_id:
                        all_nodes[node_id] = node  # IDをキーにノード情報を保存

                all_relationships = {}
                for rel in relationships:
                    rel_id = rel.get("element_id")
                    if rel_id:
                        all_relationships[rel_id] = rel

                print(
                    f"  Found {len(all_nodes)} relevant nodes and {len(all_relationships)} relationships."
                )

                # GraphRAGのコンテキストとして渡すために整形しやすい形式で返す
                # ノード、関係性それぞれのリストとして返す
                return list(all_nodes.values()) + list(
                    all_relationships.values()
                )  # ノードと関係性をまとめてリストで返す例

            else:
                print("  No matching entities found in the graph for the query.")
                return []  # 結果がない場合

    except Exception as e:
        print(f"Error during Neo4j graph search: {e}")
        return []


def format_graph_results_for_llm(graph_result: List[Dict[str, Any]]) -> str:
    """
    Neo4jの検索結果を、LLMへのプロンプトに含めるコンテキスト文字列に整形する。
    ノード情報と関係性情報を分かりやすくテキスト化する。
    """
    if not graph_result:
        return "【Graph参照情報】\nグラフから関連情報は得られませんでした。"

    context_string = "【Graph参照情報】\n"
    nodes_info = []
    relationships_info = []

    for item in graph_result:
        # ノードと関係性を区別
        if "labels" in item:  # Neo4jノードオブジェクトには 'labels' プロパティがある
            # ノード情報の整形
            name = item.get("name", "不明なノード名")
            node_type = ", ".join(
                item.get("labels", ["不明なタイプ"])
            )  # ラベルをタイプとして扱う
            properties = ", ".join(
                [
                    f"{k}: {v}"
                    for k, v in item.items()
                    if k not in ["element_id", "labels", "name", "type"]
                ]
            )  # name, type, id以外をプロパティとして表示
            source_info = (
                f" (出典: {item.get('source_file', '不明')})"
                if item.get("source_file")
                else ""
            )
            nodes_info.append(f"ノード: 「{name}」 (タイプ: {node_type}{source_info})")
            # 必要に応じて、特定のプロパティ（例: 定義、説明など）を詳細に表示しても良い

        elif (
            "type" in item and "start" in item and "end" in item
        ):  # Neo4j関係性オブジェクトには 'type', 'start', 'end' プロパティがある
            # 関係性情報の整形
            rel_type = item.get("type", "不明な関係")
            # 関係性の両端のノード名を特定するのは少し複雑になる（ノードリストを引く必要がある）
            # ここではシンプルに、関係性のタイプと、その関係性がどのノードとノードを繋いでいるか（ノードIDで）を示す
            # より良い方法としては、start/end IDを使って node_id -> node_name のマップを事前に作っておく
            # 簡単のため、関係性タイプのみ表示する例
            source_info = (
                f" (出典: {item.get('source_file', '不明')})"
                if item.get("source_file")
                else ""
            )
            relationships_info.append(f"関係性: 「{rel_type}」{source_info}")
            # より丁寧な例：
            # start_node_name = next((n['name'] for n in graph_result if 'labels' in n and n.get('element_id') == item.get('start').element_id), '不明')
            # end_node_name = next((n['name'] for n in graph_result if 'labels' in n and n.get('element_id') == item.get('end').element_id), '不明')
            # relationships_info.append(f"関係性: 「{start_node_name}」-[{rel_type}]->「{end_node_name}」{source_info}")

    if nodes_info:
        context_string += "【ノード】\n" + "\n".join(nodes_info) + "\n\n"
    if relationships_info:
        context_string += "【関係性】\n" + "\n".join(relationships_info) + "\n\n"

    return context_string.strip()


# --- メイン処理 (テスト用) ---

if __name__ == "__main__":
    # Neo4jとOllamaへの接続を試みる
    driver = None
    ollama_client = None
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("Neo4j connection successful for graph search test.")

        ollama_client = ollama.Client()
        print("Ollama client successful for query analysis.")

    except Exception as e:
        print(f"Error connecting to services for graph search test: {e}")
        if driver:
            driver.close()
        exit()  # 接続できない場合は終了

    # テスト用のクエリ
    # ステップ4aで登録したデータ（guidelines.jsonの内容など）に関連するクエリを試す
    test_query = "筑波大学のガイドラインは誰向けですか？"
    # test_query = "生成AIとは何ですか？" # 概念に関するクエリ
    # test_query = "ガイドラインの発行元はどこですか？" # 関係性に関するクエリ

    print(f'\n--- Processing Query for Graph Search: "{test_query}" ---')

    # 1. クエリからエンティティを特定
    query_entities = identify_query_entities(
        test_query, ollama_client, QUERY_ANALYSIS_MODEL
    )

    # 2. Graph検索を実行
    graph_result = perform_graph_search(query_entities, driver, MAX_HOPS)

    # 3. Graph検索結果をLLM向けに整形
    graph_context = format_graph_results_for_llm(graph_result)

    # 結果を表示
    print("\n--- Generated Graph Context for LLM ---")
    print(graph_context)
    print("\n" + "=" * 30 + "\n")

    # 処理完了後に接続を閉じる
    if driver:
        driver.close()

    print("Graph search script finished.")
