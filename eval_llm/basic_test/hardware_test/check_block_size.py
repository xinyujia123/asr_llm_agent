import urllib.request
import re
import argparse
import pdb

def fetch_kv_cache_block_size(metrics_url: str, timeout_s: int = 5) -> int:
    """
    Fetch the KV cache block size from vLLM's prometheus metrics endpoint.
    """
    try:
        print(f"Fetching metrics from {metrics_url}...")
        with urllib.request.urlopen(metrics_url, timeout=timeout_s) as response:
            metrics_text = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Failed to fetch metrics: {e}")
        return None

    # Method 1: Search line by line for metrics containing "cache" or "kv" with a "block_size" label
    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "{" not in stripped or "}" not in stripped:
            continue
        
        metric_name = stripped.split("{", 1)[0].strip().lower()
        if "cache" not in metric_name and "kv" not in metric_name:
            continue
        
        labels_part = stripped.split("{", 1)[1].split("}", 1)[0]
        
        # Parse labels
        labels = {}
        for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', labels_part):
            labels[match.group(1)] = match.group(2)
            
        block_size = labels.get("block_size")
        if block_size and block_size.isdigit():
            return int(block_size)
            
    # Method 2: Global regex fallback
    match = re.search(r'block_size="(\d+)"', metrics_text)
    if match:
        return int(match.group(1))
        
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check vLLM KV Cache Block Size")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://127.0.0.1:8000/metrics",
        help="vLLM metrics URL (default: http://127.0.0.1:8000/metrics)"
    )
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout in seconds")
    args = parser.parse_args()

    block_size = fetch_kv_cache_block_size(args.url, timeout_s=args.timeout)
    
    if block_size is not None:
        print(f"\n✅ Success: Found KV Cache block size = {block_size}")
    else:
        print("\n❌ Error: Could not find block_size in the metrics. Is vLLM running with Prometheus metrics enabled?")