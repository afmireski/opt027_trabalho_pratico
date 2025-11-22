# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "networkx",
#     "pandas",
#     "scipy",
# ]
# ///
import pandas as pd
import os
import networkx as nx
from collections import defaultdict
from itertools import combinations
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity

# Configurações
TOP_N = 100
INPUT_FILE = "datasets/merged_streaming_titles.csv"
RESULTS_DIR = "top_100_actor_graph_visualization/by_connections/results"
FILES_DIR = "top_100_actor_graph_visualization/by_connections/files"
LOUVAIN_RESOLUTION = 1.0
LOUVAIN_SEED = 42

# Leitura do dataset
print("Lendo o arquivo...")
df = pd.read_csv(INPUT_FILE)
print(f"Total de linhas no CSV: {df.shape[0]}")

# Extração e contagem de conexões entre atores
print("\nExtraindo conexões entre atores...")
actor_coactors = defaultdict(set)
pair_counts = defaultdict(int)
actor_appearances = defaultdict(int)

for cast in df['cast'].fillna(''):
    if not cast:
        continue
    
    actors = [a.strip() for a in cast.split(',') if a.strip()]
    
    # Conta aparições
    for actor in actors:
        actor_appearances[actor] += 1
    
    # Conta pares de coatores
    for a, b in combinations(actors, 2):
        key = tuple(sorted((a, b)))
        pair_counts[key] += 1
        actor_coactors[a].add(b)
        actor_coactors[b].add(a)

# Estatísticas gerais
num_actors_with_coactors = len(actor_coactors)
num_pairs = len(pair_counts)
print(f"Atores com pelo menos 1 co-ator: {num_actors_with_coactors}")
print(f"Pares únicos de atores (potenciais arestas): {num_pairs}")

# Seleção dos top 100 atores por número de conexões
top_by_connections = sorted(
    ((actor, len(neighbors)) for actor, neighbors in actor_coactors.items()), 
    key=lambda x: x[1], 
    reverse=True
)[:TOP_N]

print(f"\nTop {TOP_N} atores por número de conexões:")
for actor, connections in top_by_connections[:20]:
    appearances = actor_appearances[actor]
    print(f"{actor}: {connections} conexões, {appearances} aparições")

# Filtra apenas os top atores
top_actor_names = {actor for actor, _ in top_by_connections}

# Filtra as interações apenas entre os top 100
filtered_interactions = {
    pair: count for pair, count in pair_counts.items() 
    if pair[0] in top_actor_names and pair[1] in top_actor_names
}

print(f"\nInterações entre os top {TOP_N} atores: {len(filtered_interactions)}")
print("\nTop 10 interações mais fortes:")
for pair, count in sorted(filtered_interactions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{pair[0]} <-> {pair[1]}: {count} filmes")

# ============================================================================
# CRIAÇÃO DO GRAFO E EXTRAÇÃO DE MÉTRICAS
# ============================================================================
print("\n" + "="*60)
print("CRIANDO GRAFO E EXTRAINDO MÉTRICAS")
print("="*60)

# Função para normalizar métricas de centralidade
def normalize_centrality(centrality_dict):
    """Normaliza valores de centralidade para o intervalo [0, 1]."""
    values = list(centrality_dict.values())
    if not values:
        return centrality_dict
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 1.0 for k in centrality_dict.keys()}
    
    return {k: (v - min_val) / (max_val - min_val) for k, v in centrality_dict.items()}

# Cria o grafo do NetworkX
G = nx.Graph()

# Adiciona nós (atores)
for actor, connections in top_by_connections:
    G.add_node(actor, 
               connections=connections, 
               appearances=actor_appearances[actor])

# Adiciona arestas (interações)
for (actor1, actor2), count in filtered_interactions.items():
    G.add_edge(actor1, actor2, weight=count)

print(f"\nGrafo criado:")
print(f"  Nós (atores): {G.number_of_nodes()}")
print(f"  Arestas (conexões): {G.number_of_edges()}")

# Métricas básicas do grafo
print("\n--- MÉTRICAS BÁSICAS ---")
density = nx.density(G)
print(f"Densidade do grafo: {density:.4f}")

# Verifica conectividade
is_connected = nx.is_connected(G)
print(f"Grafo conectado: {is_connected}")

num_components = nx.number_connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
print(f"Número de componentes conectados: {num_components}")
print(f"Tamanho do maior componente: {len(largest_cc)}")

# Coeficiente de clustering
avg_clustering = nx.average_clustering(G)
print(f"Coeficiente de clustering médio: {avg_clustering:.4f}")

# Centralidade de grau
print("\n--- CENTRALIDADE DE GRAU ---")
degree_centrality = nx.degree_centrality(G)
degree_centrality_norm = normalize_centrality(degree_centrality)
top_degree = sorted(degree_centrality_norm.items(), key=lambda x: x[1], reverse=True)[:10]

# Centralidade de intermediação (betweenness)
print("\n--- CENTRALIDADE DE INTERMEDIAÇÃO ---")
betweenness_centrality = nx.betweenness_centrality(G)
betweenness_centrality_norm = normalize_centrality(betweenness_centrality)
top_betweenness = sorted(betweenness_centrality_norm.items(), key=lambda x: x[1], reverse=True)[:10]

# Centralidade de proximidade (closeness)
print("\n--- CENTRALIDADE DE PROXIMIDADE ---")
closeness_centrality = nx.closeness_centrality(G)
closeness_centrality_norm = normalize_centrality(closeness_centrality)
top_closeness = sorted(closeness_centrality_norm.items(), key=lambda x: x[1], reverse=True)[:10]

# PageRank
print("\n--- PAGERANK ---")
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
pagerank_norm = normalize_centrality(pagerank)
top_pagerank = sorted(pagerank_norm.items(), key=lambda x: x[1], reverse=True)[:10]

# Armazenar tops para salvar depois
top_centralities = {
    'top_degree': top_degree,
    'top_betweenness': top_betweenness,
    'top_closeness': top_closeness,
    'top_pagerank': top_pagerank
}

# ============================================================================
# DETECÇÃO DE COMUNIDADES (LOUVAIN)
# ============================================================================
print("\n" + "="*60)
print("DETECÇÃO DE COMUNIDADES - LOUVAIN")
print("="*60)

print(f"Executando Louvain (resolution={LOUVAIN_RESOLUTION}, seed={LOUVAIN_SEED})...")
communities_sets = louvain_communities(G, weight='weight', resolution=LOUVAIN_RESOLUTION, seed=LOUVAIN_SEED)
communities_list = [list(c) for c in communities_sets]

# Calcular modularidade
mod = modularity(G, communities_list, weight='weight')
num_communities = len(communities_list)

print(f"Modularidade: {mod:.4f}")
print(f"Número de comunidades: {num_communities}")

# Criar dicionário node -> community_id
community_map = {}
for cid, comm in enumerate(communities_list):
    for node in comm:
        community_map[node] = cid

# Estatísticas das comunidades
comm_sizes = pd.Series([len(c) for c in communities_list]).describe()
print(f"Tamanho das comunidades - min: {int(comm_sizes['min'])}, max: {int(comm_sizes['max'])}, média: {comm_sizes['mean']:.1f}")

# Criação dos diretórios
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

# Salva arquivos intermediários
print(f"\nSalvando arquivos intermediários em {FILES_DIR}...")
pd.DataFrame(
    top_by_connections, 
    columns=['Actor', 'Connections']
).to_csv(os.path.join(FILES_DIR, 'top_by_connections.csv'), index=False)

top_by_appearances = sorted(
    ((actor, actor_appearances[actor]) for actor, _ in top_by_connections),
    key=lambda x: x[1],
    reverse=True
)
pd.DataFrame(
    top_by_appearances, 
    columns=['Actor', 'Appearances']
).to_csv(os.path.join(FILES_DIR, 'top_by_appearances.csv'), index=False)

# Salva top 10 de cada métrica de centralidade
pd.DataFrame(
    top_centralities['top_degree'],
    columns=['Actor', 'Degree_Centrality']
).to_csv(os.path.join(RESULTS_DIR, 'top_degree_centrality.csv'), index=False)

pd.DataFrame(
    top_centralities['top_betweenness'],
    columns=['Actor', 'Betweenness_Centrality']
).to_csv(os.path.join(RESULTS_DIR, 'top_betweenness_centrality.csv'), index=False)

pd.DataFrame(
    top_centralities['top_closeness'],
    columns=['Actor', 'Closeness_Centrality']
).to_csv(os.path.join(RESULTS_DIR, 'top_closeness_centrality.csv'), index=False)

pd.DataFrame(
    top_centralities['top_pagerank'],
    columns=['Actor', 'PageRank']
).to_csv(os.path.join(RESULTS_DIR, 'top_pagerank.csv'), index=False)

# Salva estatísticas de comunidades
comm_stats = {
    'resolution': LOUVAIN_RESOLUTION,
    'seed': LOUVAIN_SEED,
    'modularity': mod,
    'num_communities': num_communities,
    'nodes': G.number_of_nodes(),
    'edges': G.number_of_edges()
}
pd.DataFrame([comm_stats]).to_csv(os.path.join(FILES_DIR, 'louvain_stats.csv'), index=False)

# Salva tamanhos das comunidades
comm_size_data = []
for cid, comm in enumerate(communities_list):
    comm_size_data.append({'Community_ID': cid, 'Size': len(comm)})
pd.DataFrame(comm_size_data).sort_values('Size', ascending=False).to_csv(
    os.path.join(FILES_DIR, 'community_sizes.csv'), 
    index=False
)

# Salva métricas do grafo
print("Salvando métricas do grafo...")

# Métricas consolidadas por ator
metrics_data = []
for actor in G.nodes():
    metrics_data.append({
        'Actor': actor,
        'Connections': G.nodes[actor]['connections'],
        'Appearances': G.nodes[actor]['appearances'],
        'Degree_Centrality': degree_centrality_norm[actor],
        'Betweenness_Centrality': betweenness_centrality_norm[actor],
        'Closeness_Centrality': closeness_centrality_norm[actor],
        'PageRank': pagerank_norm[actor]
    })

pd.DataFrame(metrics_data).to_csv(
    os.path.join(FILES_DIR, 'actor_metrics.csv'), 
    index=False
)

# Métricas gerais do grafo
graph_metrics = {
    'Metric': ['Nodes', 'Edges', 'Density', 'Connected', 'Avg_Clustering', 
               'Num_Components' if not is_connected else 'Diameter',
               'Largest_Component_Size' if not is_connected else 'Avg_Shortest_Path'],
    'Value': [
        G.number_of_nodes(),
        G.number_of_edges(),
        density,
        is_connected,
        avg_clustering,
        nx.number_connected_components(G) if not is_connected else nx.diameter(G),
        len(max(nx.connected_components(G), key=len)) if not is_connected else nx.average_shortest_path_length(G)
    ]
}
pd.DataFrame(graph_metrics).to_csv(
    os.path.join(FILES_DIR, 'graph_metrics.csv'), 
    index=False
)

# Gera arquivos para visualização (Flourish)
print(f"\nGerando arquivos para visualização em {RESULTS_DIR}...")

# links.csv - conexões entre atores
links_data = []
for (source, target), weight in filtered_interactions.items():
    links_data.append({
        'Source': source,
        'Target': target,
        'Value': weight
    })

links_df = pd.DataFrame(links_data)
links_file = os.path.join(RESULTS_DIR, "links.csv")
links_df.to_csv(links_file, index=False)

# points.csv - informações dos atores com comunidades e métricas
points_data = []
for actor in G.nodes():
    points_data.append({
        'id': actor,
        'Community': community_map[actor],
        'Degree': G.degree(actor),
        'Appearances': actor_appearances[actor],
        'Betweenness': round(betweenness_centrality_norm[actor], 4),
        'Closeness': round(closeness_centrality_norm[actor], 4),
        'PageRank': round(pagerank_norm[actor], 4)
    })

points_df = pd.DataFrame(points_data)
points_file = os.path.join(RESULTS_DIR, "points.csv")
points_df.to_csv(points_file, index=False)

# Análise dos top atores
print("\n--- TOP 10 ATORES POR CENTRALIDADE ---")
print("\n🔗 Degree (conexões):")
for idx, row in points_df.nlargest(10, 'Degree').iterrows():
    print(f"  {row['id']}: {row['Degree']} conexões, {row['Appearances']} aparições")

print("\n🌉 Betweenness (intermediação):")
for idx, row in points_df.nlargest(10, 'Betweenness').iterrows():
    print(f"  {row['id']}: {row['Betweenness']:.4f}")

print("\n📍 Closeness (proximidade):")
for idx, row in points_df.nlargest(10, 'Closeness').iterrows():
    print(f"  {row['id']}: {row['Closeness']:.4f}")

print("\n⭐ PageRank (importância):")
for idx, row in points_df.nlargest(10, 'PageRank').iterrows():
    print(f"  {row['id']}: {row['PageRank']:.4f}")

# Resumo final
print("\n" + "="*60)
print("RESUMO DO PROCESSAMENTO")
print("="*60)
print(f"Total de atores únicos na base: {len(actor_appearances)}")
print(f"Atores com co-atores: {num_actors_with_coactors}")
print(f"Atores selecionados (top {TOP_N}): {len(top_by_connections)}")
print(f"Total de pares únicos: {num_pairs}")
print(f"Interações entre top {TOP_N}: {len(filtered_interactions)}")
print(f"\nGrafo: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
print(f"Densidade: {density:.4f}, Clustering: {avg_clustering:.4f}")
print(f"Comunidades: {num_communities}, Modularidade: {mod:.4f}")
print(f"\nArquivos intermediários: {FILES_DIR}")
print("  - top_by_connections.csv")
print("  - top_by_appearances.csv")
print("  - top_degree_centrality.csv")
print("  - top_betweenness_centrality.csv")
print("  - top_closeness_centrality.csv")
print("  - top_pagerank.csv")
print("  - actor_metrics.csv")
print("  - graph_metrics.csv")
print("  - louvain_stats.csv")
print("  - community_sizes.csv")
print(f"\nArquivos de visualização (Flourish): {RESULTS_DIR}")
print(f"  - links.csv ({len(links_df)} arestas)")
print(f"  - points.csv ({len(points_df)} nós, {num_communities} comunidades)")
print("\n✨ Importe links.csv e points.csv no Flourish Network Graph")
print("   - Use 'id' como node identifier")
print("   - Use 'Community' para colorir os nós")
print("="*60)
