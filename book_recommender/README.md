# AI-Driven Book Recommender Chatbot

This simple project demonstrates an offline approach to recommending
books based on a text query. It loads metadata for a small sample of
books and computes TF–IDF vectors for their descriptions. When you
provide a query describing your mood, a topic, or a previous favorite,
the script computes a TF–IDF vector for your query and returns the top
five matches ranked by cosine similarity.

## Setup

No external packages are required beyond the Python standard library.

To run the recommender:

```bash
python recommender.py --query "adventure and dragons"
```

The script prints the top matching books. You can provide your own
CSV file by editing `books.csv`.
