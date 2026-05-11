# Vision token extraction (Hang)
# Loads Qwen2-VL-2B, runs all 861 videos through the frozen vision encoder,
# caches the result to vision_tokens.pt so we don't re-encode every epoch.
