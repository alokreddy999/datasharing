#!/usr/bin/env python3
"""Tiny HTTP server for spam detection accessible from an iPhone."""

from __future__ import annotations

import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from spam_classifier import NaiveBayesSpamClassifier, load_training_data


messages, labels = load_training_data()
clf = NaiveBayesSpamClassifier()
clf.fit(messages, labels)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>iPhone Spam Detector</title></head>
<body>
<h1>iPhone Spam Detector</h1>
<form method="post">
<input type="text" name="message" placeholder="Enter a message" size="50" />
<input type="submit" value="Check" />
</form>
{result}
</body>
</html>
"""


class SpamRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.respond("")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        params = urllib.parse.parse_qs(body)
        message = params.get("message", [""])[0]
        prediction = clf.predict(message) if message else ""
        result = f"<p>Prediction: {prediction}</p>" if message else ""
        self.respond(result)

    def respond(self, result: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        page = HTML_TEMPLATE.format(result=result)
        self.wfile.write(page.encode())


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    with HTTPServer((host, port), SpamRequestHandler) as httpd:
        print(f"Serving on http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
