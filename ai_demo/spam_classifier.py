#!/usr/bin/env python3
"""Simple Naive Bayes spam classifier example."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple


class NaiveBayesSpamClassifier:
    def __init__(self) -> None:
        self.class_word_counts: Dict[str, defaultdict[str, int]] = {}
        self.class_totals: Dict[str, int] = {}
        self.class_priors: Dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, messages: List[str], labels: List[str]) -> None:
        """Train the classifier with messages and their labels."""
        self.class_word_counts = {"spam": defaultdict(int), "ham": defaultdict(int)}
        self.class_totals = {"spam": 0, "ham": 0}
        self.class_priors = {"spam": 0.0, "ham": 0.0}

        total_messages = len(messages)
        spam_messages = sum(1 for label in labels if label == "spam")
        ham_messages = total_messages - spam_messages
        self.class_priors["spam"] = spam_messages / total_messages
        self.class_priors["ham"] = ham_messages / total_messages

        for message, label in zip(messages, labels):
            words = self._tokenize(message)
            for word in words:
                self.class_word_counts[label][word] += 1
                self.class_totals[label] += 1
                self.vocab.add(word)

    def predict(self, message: str) -> str:
        """Classify a message as spam or ham."""
        words = self._tokenize(message)
        spam_score = math.log(self.class_priors["spam"]) if self.class_priors["spam"] else float('-inf')
        ham_score = math.log(self.class_priors["ham"]) if self.class_priors["ham"] else float('-inf')

        for word in words:
            spam_score += math.log(
                (self.class_word_counts["spam"].get(word, 0) + 1)
                / (self.class_totals["spam"] + len(self.vocab))
            )
            ham_score += math.log(
                (self.class_word_counts["ham"].get(word, 0) + 1)
                / (self.class_totals["ham"] + len(self.vocab))
            )

        return "spam" if spam_score > ham_score else "ham"

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())


def load_training_data() -> Tuple[List[str], List[str]]:
    messages = [
        "Congratulations! You've won a free ticket.",
        "Reminder: our meeting is at 10am tomorrow.",
        "Earn dollars quickly, click this link now!",
        "Lunch at noon?",
        "Hot singles in your area are waiting.",
        "Project deadline is next week.",
        "Get cheap meds without prescription.",
        "Are we still on for dinner tonight?",
        "Win a new iPhone by entering here!",
        "Don't forget to submit the report.",
    ]
    labels = [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
    ]
    return messages, labels


def demo() -> None:
    print("Training simple spam classifier...")
    messages, labels = load_training_data()
    clf = NaiveBayesSpamClassifier()
    clf.fit(messages, labels)
    print("Classifier ready! Type a message and press enter (or just press enter to exit).")
    while True:
        try:
            user_input = input("Message: ").strip()
        except EOFError:
            break
        if not user_input:
            break
        prediction = clf.predict(user_input)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    demo()
