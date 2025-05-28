# graph_search.py

import json
from typing import Any, Dict, List

import ollama
from neo4j import GraphDatabase

# --- 設定 ---
MAX_HOPS = 2  # グラフ検索の最大ホップ数

# --- ヘルパー関数 ---


def identify_query_entities(
    query: str, ollama_client: ollama.Client, model_name: str
) -> List[str]:
    """
    Ollamaモデルを使用してクエリから主要なエンティティ名を特定する。
    """
    prompt = f"""
    以下の質問文から、Neo4jデータベースに存在する可能性のある、具体的なエンティティ名を抽出してください。
    抽出するエンティティは、組織名、人名、具体的な概念、文書要素など、質問の核心となる固有名詞やキーワードです。
    抽出されたエンティティ名は、それぞれが独立した文字列として、Pythonのリスト形式で出力してください。
    エンティティが見つからない場合は、空のリスト `[]` を返してください。

    例1:
    質問: 筑波大学の生成AIガイドラインについて教えてください。
    出力: ["筑波大学", "生成AIガイドライン"]

    例2:
    質問: 大学のルールについて教えてください。
    出力: []

    質問: {query}
    出力:
    """

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},  # 厳密な抽出のため温度は低めに
        )
        response_content = response["message"]["content"]

        # 修正: 応答テキストからリスト形式を抽出するロジックを強化
        # 単純なJSONリストとしてパースを試みる
        parsed_list = json.loads(response_content)
        if isinstance(parsed_list, list) and all(
            isinstance(item, str) for item in parsed_list
        ):
            return parsed_list
        else:
            print(
                f"Ollama returned non-list or non-string list for entity identification: {response_content}"
            )
            return []

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in identify_query_entities: {e}")
        print(f"Problematic content: {response_content}")
        return []
    except Exception as e:
        print(f"Error during query entity identification with Ollama: {e}")
        return []


def perform_graph_search(
    query_entity_names: List[str], driver: GraphDatabase.driver, max_hops: int = 2
) -> List[Dict[str, Any]]:
    """
    特定されたエンティティ名を基にNeo4jグラフを検索し、関連ノードとエッジを返す。
    """
    results = []
    with driver.session() as session:
        for entity_name in query_entity_names:
            print(f"  Searching graph for entity: '{entity_name}'")
            # 修正: Cypherクエリのパターンを調整
            # 関連するノードと関係性を取得し、ノードのプロパティを含める
            # 検索結果をJSON形式で返すのではなく、Neo4jドライバーのRecordとして直接扱う
            query_string = f"""
                MATCH (start:Entity {{name: $entityName}})
                MATCH (start)-[r*1..{max_hops}]-(related:Entity)
                RETURN start, r, related
            """
            try:
                # `data()` を使用して結果をPythonの辞書形式で取得
                query_results = session.run(query_string, entityName=entity_name).data()

                # 結果を整形してresultsリストに追加
                for record in query_results:
                    # Neo4jのRecordオブジェクトから直接プロパティとタイプを取得
                    # startノード
                    start_node = record["start"]
                    start_node_dict = {
                        "type": "node",
                        "labels": list(start_node.labels),
                        "properties": dict(start_node),
                    }

                    # relatedノード
                    related_node = record["related"]
                    related_node_dict = {
                        "type": "node",
                        "labels": list(related_node.labels),
                        "properties": dict(related_node),
                    }

                    # r (pathオブジェクトまたはリスト)
                    # パスに含まれるすべての関係性とノードを分解して追加
                    path_elements = []
                    if "r" in record:
                        # 'r' がPathオブジェクトの場合、関係性とノードを抽出
                        # r はリストとして返される可能性もあるので、対応
                        relationships_in_path = record["r"]
                        if not isinstance(relationships_in_path, list):
                            relationships_in_path = [
                                relationships_in_path
                            ]  # 単一のリレーションの場合もリストとして扱う

                        for rel in relationships_in_path:
                            rel_dict = {
                                "type": "relationship",
                                "rel_type": rel.type,
                                "start_node_name": rel.start_node[
                                    "name"
                                ],  # 関係の開始ノード
                                "end_node_name": rel.end_node[
                                    "name"
                                ],  # 関係の終了ノード
                                "properties": dict(rel),
                            }
                            path_elements.append(rel_dict)
                            # 関係の開始ノードと終了ノードも追加 (重複は後で除去)
                            # path_elements.append({'type': 'node', 'labels': list(rel.start_node.labels), 'properties': dict(rel.start_node)})
                            # path_elements.append({'type': 'node', 'labels': list(rel.end_node.labels), 'properties': dict(rel.end_node)})

                    results.append(
                        {
                            "start_node": start_node_dict,
                            "related_node": related_node_dict,
                            "path_relationships": path_elements,
                        }
                    )

                print(f"  Found {len(query_results)} paths for '{entity_name}'.")

            except Exception as e:
                print(f"Error executing graph search for '{entity_name}': {e}")
    return results


def format_graph_results_for_llm(graph_results: List[Dict[str, Any]]) -> str:
    """
    Graph検索結果をLLMに渡すための整形されたコンテキスト文字列に変換する。
    """
    if not graph_results:
        return "【Graph参照情報】\nGraph検索では関連情報が見つかりませんでした。"

    context_str = "【Graph参照情報】\n"

    # 重複を避けるためのセット
    seen_nodes = set()
    seen_relationships = set()

    # まずノードのリストを作成
    nodes_list = []
    for record in graph_results:
        # 開始ノード
        start_node_name = record["start_node"]["properties"].get("name", "不明なノード")
        if start_node_name not in seen_nodes:
            nodes_list.append(
                f"- ノード: {start_node_name} (タイプ: {record['start_node']['properties'].get('type', 'N/A')}, ファイル: {record['start_node']['properties'].get('source_file', 'N/A')})"
            )
            seen_nodes.add(start_node_name)

        # 関連ノード
        related_node_name = record["related_node"]["properties"].get(
            "name", "不明なノード"
        )
        if related_node_name not in seen_nodes:
            nodes_list.append(
                f"- ノード: {related_node_name} (タイプ: {record['related_node']['properties'].get('type', 'N/A')}, ファイル: {record['related_node']['properties'].get('source_file', 'N/A')})"
            )
            seen_nodes.add(related_node_name)

    if nodes_list:
        context_str += "--- 関連ノード ---\n"
        context_str += "\n".join(nodes_list) + "\n"

    # 次に関係性のリストを作成
    relationships_list = []
    for record in graph_results:
        if "path_relationships" in record:
            for rel in record["path_relationships"]:
                # 関係性のユニークな識別子を作成 (例: type + start_node_name + end_node_name)
                rel_id = (
                    f"{rel['rel_type']}_{rel['start_node_name']}_{rel['end_node_name']}"
                )
                if rel_id not in seen_relationships:
                    relationships_list.append(
                        f"- 関係性: ({rel['start_node_name']}) -[:{rel['rel_type']}]-> ({rel['end_node_name']}) "
                        f"(ファイル: {rel['properties'].get('source_file', 'N/A')})"
                    )
                    seen_relationships.add(rel_id)

    if relationships_list:
        context_str += "--- 関連関係性 ---\n"
        context_str += "\n".join(relationships_list) + "\n"

    # ノード数と関係性数の概要を追加
    num_nodes = len(seen_nodes)
    num_rels = len(seen_relationships)
    context_str += f"概要: ノード: {num_nodes}, 関係性: {num_rels} 取得\n"

    return context_str


# --- メイン処理関数 (GraphRAG検索とコンテキスト取得) ---
def get_graph_context(
    query: str, neo4j_driver: GraphDatabase.driver, ollama_client: ollama.Client
) -> str:
    """
    ユーザーのクエリに対してGraph検索を実行し、LLMに渡すコンテキストを返す。
    """
    if neo4j_driver is None:
        return "【Graph参照情報】\nNeo4jドライバーが利用できません。"

    # 1. クエリからエンティティを特定 (Ollama利用)
    # QUERY_ANALYSIS_MODEL は app.py から渡されるか、ここで別途定義
    query_entities = identify_query_entities(
        query, ollama_client, "mistral"
    )  # app.pyと合わせる

    if not query_entities:
        return "【Graph参照情報】\nクエリからGraph検索の起点となるエンティティを特定できませんでした。"

    # 2. Graph検索を実行
    graph_result = perform_graph_search(query_entities, neo4j_driver, MAX_HOPS)

    # 3. Graph検索結果をLLM向けに整形
    graph_context = format_graph_results_for_llm(graph_result)

    # 修正: 整形されたコンテキスト文字列を返す
    return graph_context


# --- スクリプトとして直接実行する部分 ---
if __name__ == "__main__":
    # このブロックはテスト用です。
    # app.py は get_graph_context 関数を直接呼び出します。

    # テスト用の設定（環境変数から読み込むか、直接指定）
    OLLAMA_HOST_TEST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    NEO4J_URI_TEST = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER_TEST = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD_TEST = os.getenv(
        "NEO4J_PASSWORD", "password"
    )  # YOUR_NEO4J_PASSWORD_HERE

    # Ollamaクライアントの初期化
    try:
        test_ollama_client = ollama.Client(host=OLLAMA_HOST_TEST)
        test_ollama_client.version()
        print(
            f"Test Ollama client connected successfully. Version: {test_ollama_client.version()}"
        )
    except Exception as e:
        print(
            f"Test Ollama client connection failed: {e}. Ensure Ollama is running at {OLLAMA_HOST_TEST}"
        )
        exit()

    # Neo4jドライバーの初期化
    test_driver = None
    try:
        test_driver = GraphDatabase.driver(
            NEO4J_URI_TEST, auth=(NEO4J_USER_TEST, NEO4J_PASSWORD_TEST)
        )
        test_driver.verify_connectivity()
        print("Test Neo4j driver connected successfully.")
    except Exception as e:
        print(
            f"Test Neo4j connection failed: {e}. Ensure Neo4j is running at {NEO4J_URI_TEST}"
        )
        exit()

    test_query = "筑波大学の生成AIガイドラインの内容について教えてください。"
    print(f"\nTest Query: {test_query}")

    graph_context = get_graph_context(test_query, test_driver, test_ollama_client)
    print("\n--- Generated GraphRAG Context ---")
    print(graph_context)
    print("----------------------------------")

    if test_driver:
        test_driver.close()
