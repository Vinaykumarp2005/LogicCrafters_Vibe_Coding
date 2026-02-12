"""
SEFS Semantic Engine
Performs semantic analysis, embedding generation, and clustering of file contents.
Uses TF-IDF (lightweight) or Sentence Transformers (accurate) for embeddings,
and automatic K-means clustering with silhouette-score-based cluster count selection.
"""

import logging
import re
import numpy as np
from collections import Counter
from pathlib import Path

logger = logging.getLogger("sefs.semantic_engine")


class SemanticEngine:
    """Computes semantic embeddings and clusters files by content similarity."""

    def __init__(self, method="tfidf", model_name="all-MiniLM-L6-v2",
                 num_clusters=0, min_clusters=2, max_clusters=10,
                 similarity_threshold=0.15):
        self.method = method
        self.model_name = model_name
        self.num_clusters = num_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold

        self._embeddings = {}      # path -> embedding vector
        self._file_data = {}       # path -> file metadata dict
        self._clusters = {}        # cluster_id -> {"label": str, "files": [paths]}
        self._file_to_cluster = {} # path -> cluster_id

        # TF-IDF components
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        # Transformer model (lazy-loaded)
        self._transformer_model = None

        logger.info(f"SemanticEngine initialized with method={method}")

    def _get_transformer_model(self):
        """Lazy-load the sentence transformer model."""
        if self._transformer_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading transformer model: {self.model_name}")
                self._transformer_model = SentenceTransformer(self.model_name)
                logger.info("Transformer model loaded successfully")
            except ImportError:
                logger.error("sentence-transformers not installed. Falling back to TF-IDF.")
                self.method = "tfidf"
                return None
        return self._transformer_model

    def process_files(self, file_data_list: list) -> dict:
        """
        Process a list of file data dicts and compute clusters.

        Args:
            file_data_list: list of dicts with at least 'path' and 'content' keys

        Returns:
            dict of cluster_id -> {"label": str, "files": [file_data_dicts]}
        """
        # Filter files with actual content
        valid_files = [f for f in file_data_list if f.get("content", "").strip()]

        if not valid_files:
            logger.warning("No files with extractable content to process")
            self._clusters = {}
            self._file_to_cluster = {}
            return self._clusters

        # Store file data
        for f in valid_files:
            self._file_data[f["path"]] = f

        # Compute embeddings
        if self.method == "transformer":
            embeddings = self._compute_transformer_embeddings(valid_files)
        else:
            embeddings = self._compute_tfidf_embeddings(valid_files)

        if embeddings is None or len(embeddings) == 0:
            return self._clusters

        # Store embeddings
        for i, f in enumerate(valid_files):
            self._embeddings[f["path"]] = embeddings[i]

        # Cluster files
        self._clusters = self._cluster_files(valid_files, embeddings)

        return self._clusters

    def _compute_tfidf_embeddings(self, file_data_list: list) -> np.ndarray:
        """Compute TF-IDF embeddings for file contents."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            contents = [f["content"] for f in file_data_list]

            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )

            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(contents)
            embeddings = self._tfidf_matrix.toarray()

            logger.info(f"TF-IDF embeddings computed: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"TF-IDF embedding error: {e}")
            return None

    def _compute_transformer_embeddings(self, file_data_list: list) -> np.ndarray:
        """Compute sentence-transformer embeddings for file contents."""
        model = self._get_transformer_model()
        if model is None:
            return self._compute_tfidf_embeddings(file_data_list)

        try:
            # Truncate content to first 512 tokens worth of text for efficiency
            contents = []
            for f in file_data_list:
                text = f["content"][:2000]  # ~512 tokens
                contents.append(text)

            embeddings = model.encode(contents, show_progress_bar=False,
                                       normalize_embeddings=True)
            embeddings = np.array(embeddings)

            logger.info(f"Transformer embeddings computed: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Transformer embedding error: {e}")
            return self._compute_tfidf_embeddings(file_data_list)

    def _cluster_files(self, file_data_list: list, embeddings: np.ndarray) -> dict:
        """Cluster files based on embeddings using K-Means with auto-K."""
        n_files = len(file_data_list)

        if n_files <= 1:
            # Single file - put in one cluster
            cluster = {
                0: {
                    "label": self._generate_cluster_label(file_data_list, [0]),
                    "files": [f["path"] for f in file_data_list],
                }
            }
            for f in file_data_list:
                self._file_to_cluster[f["path"]] = 0
            return cluster

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import normalize

            # Normalize embeddings
            embeddings_norm = normalize(embeddings)

            # Determine optimal number of clusters
            if self.num_clusters > 0:
                optimal_k = min(self.num_clusters, n_files)
            else:
                optimal_k = self._find_optimal_k(embeddings_norm, n_files)

            logger.info(f"Clustering {n_files} files into {optimal_k} clusters")

            # Perform clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42,
                          n_init=10, max_iter=300)
            labels = kmeans.fit_predict(embeddings_norm)

            # Build cluster dict
            clusters = {}
            for i, label in enumerate(labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = {"files": [], "indices": []}
                clusters[label]["files"].append(file_data_list[i]["path"])
                clusters[label]["indices"].append(i)
                self._file_to_cluster[file_data_list[i]["path"]] = label

            # Generate labels for each cluster
            for cid in clusters:
                indices = clusters[cid]["indices"]
                cluster_files = [file_data_list[i] for i in indices]
                clusters[cid]["label"] = self._generate_cluster_label(cluster_files, indices)
                del clusters[cid]["indices"]  # Clean up temp data

            self._clusters = clusters
            return clusters

        except Exception as e:
            logger.error(f"Clustering error: {e}")
            # Fallback: everything in one cluster
            cluster = {
                0: {
                    "label": "All Files",
                    "files": [f["path"] for f in file_data_list],
                }
            }
            for f in file_data_list:
                self._file_to_cluster[f["path"]] = 0
            return cluster

    def _find_optimal_k(self, embeddings: np.ndarray, n_files: int) -> int:
        """Find optimal number of clusters using silhouette score."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        min_k = max(self.min_clusters, 2)
        max_k = min(self.max_clusters, n_files - 1, 10)

        if min_k >= n_files:
            return max(1, n_files // 2)

        if max_k < min_k:
            return min_k

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                # Need at least 2 unique labels for silhouette score
                if len(set(labels)) < 2:
                    continue

                score = silhouette_score(embeddings, labels)
                logger.debug(f"  k={k}, silhouette={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        logger.info(f"Optimal k={best_k} (silhouette={best_score:.4f})")
        return best_k

    def _generate_cluster_label(self, cluster_files: list, indices: list) -> str:
        """Generate a human-readable label for a cluster based on content keywords."""
        try:
            # Combine all content
            combined = " ".join(f.get("content", "")[:500] for f in cluster_files)

            # Extract keywords using simple frequency analysis
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined.lower())

            # Remove common stop words
            stop_words = {
                "the", "and", "for", "are", "but", "not", "you", "all",
                "can", "had", "her", "was", "one", "our", "out", "has",
                "have", "each", "make", "like", "been", "from", "this",
                "that", "with", "they", "will", "would", "there", "their",
                "what", "about", "which", "when", "were", "them", "than",
                "its", "into", "more", "other", "some", "such", "also",
                "use", "used", "using", "how", "may", "these", "just",
                "over", "very", "then", "only", "come", "could", "being",
                "any", "after", "most", "should", "does", "did", "get",
                "got", "way", "well", "here", "where", "why", "who",
            }

            filtered = [w for w in words if w not in stop_words]
            freq = Counter(filtered)

            # Get top 2-3 keywords
            top_keywords = [word for word, _ in freq.most_common(3)]

            if top_keywords:
                label = " & ".join(kw.capitalize() for kw in top_keywords[:2])
                return label
            else:
                # Fallback to file extensions or generic name
                exts = set(f.get("extension", "") for f in cluster_files)
                if exts:
                    return f"Files ({', '.join(exts)})"
                return "Miscellaneous"

        except Exception as e:
            logger.warning(f"Label generation error: {e}")
            return "Unnamed Group"

    def get_clusters(self) -> dict:
        """Return current cluster assignments."""
        return self._clusters

    def get_file_cluster(self, file_path: str) -> int:
        """Get the cluster ID for a specific file."""
        return self._file_to_cluster.get(file_path, -1)

    def get_similarity_matrix(self) -> dict:
        """Compute pairwise similarity matrix for visualization."""
        if not self._embeddings:
            return {"files": [], "matrix": []}

        from sklearn.metrics.pairwise import cosine_similarity

        paths = list(self._embeddings.keys())
        vectors = np.array([self._embeddings[p] for p in paths])

        sim_matrix = cosine_similarity(vectors)

        return {
            "files": paths,
            "matrix": sim_matrix.tolist()
        }

    def get_graph_data(self) -> dict:
        """
        Generate graph data for the 2D visualization.
        Returns nodes and edges suitable for D3.js force-directed layout.
        """
        nodes = []
        edges = []
        seen_paths = set()

        # Create nodes
        for cid, cluster_data in self._clusters.items():
            for file_path in cluster_data["files"]:
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)
                file_info = self._file_data.get(file_path, {})
                nodes.append({
                    "id": file_path,
                    "name": Path(file_path).name,
                    "cluster": cid,
                    "cluster_label": cluster_data["label"],
                    "size": file_info.get("size", 0),
                    "extension": file_info.get("extension", ""),
                    "modified": file_info.get("modified", ""),
                })

        # Create edges based on similarity
        if len(self._embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity

            paths = list(self._embeddings.keys())
            vectors = np.array([self._embeddings[p] for p in paths])
            sim_matrix = cosine_similarity(vectors)

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    similarity = float(sim_matrix[i][j])
                    if similarity > self.similarity_threshold:
                        edges.append({
                            "source": paths[i],
                            "target": paths[j],
                            "weight": similarity,
                        })

        return {"nodes": nodes, "edges": edges}

    def remove_file(self, file_path: str):
        """Remove a file from the semantic engine's state."""
        self._embeddings.pop(file_path, None)
        self._file_data.pop(file_path, None)
        self._file_to_cluster.pop(file_path, None)
        # Clean up cluster references
        for cid in list(self._clusters.keys()):
            if file_path in self._clusters[cid]["files"]:
                self._clusters[cid]["files"].remove(file_path)
                if not self._clusters[cid]["files"]:
                    del self._clusters[cid]

    def update_paths(self, path_mapping: dict):
        """
        Update internal paths after files are moved by the folder manager.
        path_mapping: dict of old_path -> new_path
        """
        # Update embeddings
        for old_path, new_path in path_mapping.items():
            if old_path in self._embeddings:
                self._embeddings[new_path] = self._embeddings.pop(old_path)
            if old_path in self._file_data:
                self._file_data[new_path] = self._file_data.pop(old_path)
                self._file_data[new_path]["path"] = new_path
                self._file_data[new_path]["name"] = Path(new_path).name
            if old_path in self._file_to_cluster:
                cid = self._file_to_cluster.pop(old_path)
                self._file_to_cluster[new_path] = cid

        # Update cluster file lists
        for cid in self._clusters:
            new_files = []
            for fp in self._clusters[cid]["files"]:
                new_files.append(path_mapping.get(fp, fp))
            self._clusters[cid]["files"] = new_files

        logger.info(f"Updated {len(path_mapping)} file paths in semantic engine")
