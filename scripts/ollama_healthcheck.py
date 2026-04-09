#!/usr/bin/env python3

import sys
import urllib.error
import urllib.request


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 0 if 200 <= response.status < 300 else 1
    except (urllib.error.URLError, TimeoutError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
