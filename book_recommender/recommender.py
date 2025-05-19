import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Book:
    title: str
    author: str
    description: str
    vector: Counter


class BookRecommender:
    """Book recommender using a simple TF-IDF representation."""

    def __init__(self, csv_path: str) -> None:
        self.books: List[Book] = []
        self.vocabulary: set[str] = set()
        descriptions = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                tokens = tokenize(row["description"])
                desc_counter = Counter(tokens)
                self.books.append(Book(row["title"], row["author"], row["description"], desc_counter))
                self.vocabulary.update(desc_counter.keys())
                descriptions.append(tokens)

        self.idf = self._compute_idf(descriptions)
        self.book_vectors = [self._tfidf_vector(book.vector) for book in self.books]

    def _compute_idf(self, docs: List[List[str]]):
        N = len(docs)
        df = Counter()
        for doc in docs:
            df.update(set(doc))
        idf = {term: math.log((N + 1) / (df[term] + 1)) + 1 for term in self.vocabulary}
        return idf

    def _tfidf_vector(self, term_counts: Counter):
        vector = {}
        for term, count in term_counts.items():
            if term in self.idf:
                vector[term] = count * self.idf[term]
        return vector

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        intersection = set(vec1) & set(vec2)
        num = sum(vec1[t] * vec2[t] for t in intersection)
        denom1 = math.sqrt(sum(v * v for v in vec1.values()))
        denom2 = math.sqrt(sum(v * v for v in vec2.values()))
        if denom1 == 0 or denom2 == 0:
            return 0.0
        return num / (denom1 * denom2)

    def recommend(self, query: str, top_k: int = 5):
        query_tokens = tokenize(query)
        query_counts = Counter(query_tokens)
        query_vec = self._tfidf_vector(query_counts)
        similarities = [self._cosine_similarity(query_vec, vec) for vec in self.book_vectors]
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(
            self.books[idx].title,
            self.books[idx].author,
            score,
        ) for idx, score in ranked]
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple book recommender")
    parser.add_argument("--query", required=True, help="Query describing mood, topic, or favorite book")
    parser.add_argument("--data", default="books.csv", help="Path to CSV containing books data")
    args = parser.parse_args()

    csv_path = Path(args.data)
    recommender = BookRecommender(str(csv_path))
    results = recommender.recommend(args.query)
    for title, author, score in results:
        print(f"{title} by {author} (score={score:.3f})")


if __name__ == "__main__":
    main()
