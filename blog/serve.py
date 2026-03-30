#!/usr/bin/env python3
"""
Minimal local server for the blog post.

Reads index.md, converts it to HTML with github-markdown-css styling,
and serves everything (including images via relative paths) on localhost:8000.

Usage:
    python serve.py          # serves on http://localhost:8000
    python serve.py 9000     # serves on http://localhost:9000

Dependencies:
    pip install markdown     # or: uv add markdown
"""

import http.server
import os
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Missing 'markdown' package. Install it with:")
    print("  pip install markdown")
    print("  # or: uv add markdown")
    sys.exit(1)

BLOG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BLOG_DIR.parent

# ---------------------------------------------------------------------------
#  Markdown -> HTML rendering
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown-light.min.css"
      integrity="sha512-Pmhg2i/F7+5+7SsdoUqKeH7UAZoVMYb9sXN7MbJ3QAsJSdikq96v0I2FGujmFHBXfT18DORqHAz2FVOTqYyFw=="
      crossorigin="anonymous" referrerpolicy="no-referrer" />
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
      crossorigin="anonymous" />
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
        crossorigin="anonymous"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        crossorigin="anonymous"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]
        }});"></script>
<style>
    body {{
        max-width: 860px;
        margin: 40px auto;
        padding: 0 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
                     Arial, sans-serif;
        background: #fff;
    }}
    .markdown-body {{
        box-sizing: border-box;
        min-width: 200px;
        max-width: 860px;
        margin: 0 auto;
    }}
    .markdown-body img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1.5em auto;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
    }}
    .markdown-body table {{
        display: table;
        width: 100%;
    }}
    .markdown-body table th,
    .markdown-body table td {{
        padding: 8px 16px;
    }}
</style>
</head>
<body>
<article class="markdown-body">
{body}
</article>
</body>
</html>
"""


def render_markdown(md_path: Path) -> str:
    """Read a markdown file and return complete HTML."""
    md_text = md_path.read_text(encoding="utf-8")

    # Convert image paths: ../data/v2/results/X.png -> /images/X.png
    md_text = re.sub(
        r"\.\./data/v2/results/([^\s\)]+)",
        r"/images/\1",
        md_text,
    )

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    # Extract title from first <h1>
    title_match = re.search(r"<h1>(.*?)</h1>", body)
    title = title_match.group(1) if title_match else "Blog Post"

    return HTML_TEMPLATE.format(title=title, body=body)


# ---------------------------------------------------------------------------
#  HTTP Server
# ---------------------------------------------------------------------------

class BlogHandler(http.server.BaseHTTPRequestHandler):
    """Serves the rendered blog post and images."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_blog()
        elif self.path.startswith("/images/"):
            self._serve_image()
        else:
            self.send_error(404)

    def _serve_blog(self):
        md_path = BLOG_DIR / "index.md"
        if not md_path.exists():
            self.send_error(404, "index.md not found")
            return

        html = render_markdown(md_path)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_image(self):
        # Map /images/foo.png -> PROJECT_ROOT/data/v2/results/foo.png
        filename = self.path.replace("/images/", "", 1)
        img_path = PROJECT_ROOT / "data" / "v2" / "results" / filename

        if not img_path.exists() or not img_path.is_file():
            self.send_error(404, f"Image not found: {filename}")
            return

        # Determine content type
        suffix = img_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(img_path.stat().st_size))
        self.end_headers()
        self.wfile.write(img_path.read_bytes())

    def log_message(self, format, *args):
        """Quieter logging."""
        print(f"  {args[0]}")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    # Verify index.md exists
    md_path = BLOG_DIR / "index.md"
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        sys.exit(1)

    # Verify images exist
    img_dir = PROJECT_ROOT / "data" / "v2" / "results"
    expected = ["conditional_sampling.png", "reconstruction_quality.png", "progress_comparison.png"]
    for name in expected:
        p = img_dir / name
        if p.exists():
            print(f"  [ok] {p.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [!!] Missing: {p.relative_to(PROJECT_ROOT)}")

    server = http.server.HTTPServer(("localhost", port), BlogHandler)
    print(f"\nServing blog at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
