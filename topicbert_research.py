# ==========================================================
# TOPICBERT (Research-level version, adapted for GitHub Issues)
# Based on Asgari-Chenaghlu et al. (2020)
# ==========================================================

import json
import math
import networkx as nx
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------
#  LOAD DATA
# ----------------------------------ini------------------------
df = pd.read_csv("issues_unlabeled.csv")
print(f"Total issues (unlabelled): {len(df)}")

if "title" in df.columns and "body" in df.columns:
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
else:
    raise ValueError("Dataset harus memiliki kolom 'title' dan 'body'.")

print(f"Total issues: {len(df)}")

# Buat kolom waktu simulasi (anggap urutan issue = waktu)
df["timestamp"] = range(1, len(df) + 1)

# ----------------------------------------------------------
#  TRANSFORMER (BERT EMBEDDING)
# ----------------------------------------------------------
print("\nMenghitung embedding dengan Sentence-BERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")
df["embedding"] = df["text"].apply(lambda x: model.encode(x))

# ----------------------------------------------------------
#  MEMORY GRAPH (MENGINGAT DAN MELUPAKAN)
# ----------------------------------------------------------
G = nx.Graph()
tau = 200  # tingkat “daya ingat” (semakin besar = makin lama diingat)

def add_to_memory_graph(G, text, timestamp):
    """Masukkan teks sebagai subgraf baru ke Memory Graph"""
    words = [w.lower() for w in text.split() if len(w) > 2]
    for i, w1 in enumerate(words):
        for w2 in words[i + 1 :]:
            if w1 == w2:
                continue
            if not G.has_edge(w1, w2):
                G.add_edge(w1, w2, weight=1.0, last_seen=timestamp)
            else:
                G[w1][w2]["weight"] += 1
                G[w1][w2]["last_seen"] = timestamp

def apply_forgetting(G, current_time, tau=200):
    """Gunakan rumus Ebbinghaus untuk menurunkan bobot edge lama"""
    for u, v, data in list(G.edges(data=True)):
        t = current_time - data["last_seen"]
        decay = math.exp(-t / tau)
        G[u][v]["weight"] *= decay
        if G[u][v]["weight"] < 0.01:
            G.remove_edge(u, v)

# ----------------------------------------------------------
#  GRAPH UPDATE RULES (U, I, M, S)
# ----------------------------------------------------------
clusters = {}  # {id_cluster: set_of_words}

def update_clusters(G, text):
    words = set(text.lower().split())
    matched_clusters = [
        cid for cid, words_c in clusters.items() if len(words_c & words) > 2
    ]

    if not matched_clusters:
        # U: Unique → buat cluster baru
        clusters[len(clusters)] = words
    elif len(matched_clusters) == 1:
        # I: Incessant → update cluster yang sama
        clusters[matched_clusters[0]].update(words)
    else:
        # M/S: Multiple / Subset → gabung ke cluster dengan kemiripan tertinggi
        target = max(matched_clusters, key=lambda c: len(clusters[c] & words))
        clusters[target].update(words)

# ----------------------------------------------------------
#  PROSES DATA SECARA BERTAHAP (SIMULASI STREAMING)
# ----------------------------------------------------------
for t, issue in enumerate(df.itertuples(), start=1):
    add_to_memory_graph(G, issue.text, t)
    update_clusters(G, issue.text)
    apply_forgetting(G, t, tau)
    if t % 500 == 0:
        print(f"Processed {t} issues... Memory Graph edges: {G.number_of_edges()}")

# ----------------------------------------------------------
#  TOPIC EXTRACTION (SCORING & KEYWORDS)
# ----------------------------------------------------------
print("\n🔍 Mengekstrak kata penting dari setiap cluster...")

topic_keywords = {}
vectorizer = TfidfVectorizer(stop_words="english", max_features=20)

for cid, words_c in clusters.items():
    subset_texts = [" ".join(words_c)]
    try:
        X = vectorizer.fit_transform(subset_texts)
        words = vectorizer.get_feature_names_out()
        topic_keywords[cid] = list(words)
    except ValueError:
        topic_keywords[cid] = list(words_c)[:10]

print(f"Total topik terdeteksi: {len(topic_keywords)}")

# ----------------------------------------------------------
#  VISUALISASI MEMORY GRAPH
# ----------------------------------------------------------
print("\nMenampilkan visualisasi Memory Graph...")
plt.figure(figsize=(10, 8))

# simpan layout pakai algoritma ringan
pos = nx.kamada_kawai_layout(G)  # jauh lebih cepat daripada spring_layout

node_sizes = [min(800, G.degree(n) * 10) for n in G.nodes()]
nx.draw(
    G,
    pos,
    node_color="skyblue",
    node_size=node_sizes,
    with_labels=False,
    edge_color="gray",
)
plt.title("TopicBERT Memory Graph (GitHub Issues)", fontsize=14)
plt.axis("off")

# langsung simpan hasilnya ke file
plt.savefig("memory_graph.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n Memory Graph disimpan ke 'memory_graph.png'")

nx.write_gexf(G, "memory_graph.gexf")
print(" Struktur graf juga disimpan ke 'memory_graph.gexf' (bisa dibuka di Gephi)")

# ----------------------------------------------------------
#  SIMPAN HASIL
# ----------------------------------------------------------
topics_df = pd.DataFrame([
    {"topic_id": cid, "keywords": ", ".join(words)}
    for cid, words in topic_keywords.items()
])

topics_df.to_csv("topicbert_research.csv", index=False)
print("\n Hasil disimpan di 'topicbert_research.csv'")
