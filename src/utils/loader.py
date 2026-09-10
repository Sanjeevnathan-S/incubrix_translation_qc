import json
from pathlib import Path
from typing import List, Dict, Any

def load_input_segments(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads text segments from a .json or .jsonl file while preserving segment IDs 
    and existing metadata[cite: 1].
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    segments = []
    
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item.setdefault("id", f"seg_{line_no}")
                    segments.append(item)
                    
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    if isinstance(item, str):
                        item = {"id": f"seg_{idx}", "text": item}
                    else:
                        item.setdefault("id", f"seg_{idx}")
                    segments.append(item)
            elif isinstance(data, dict):
                data.setdefault("id", "seg_1")
                segments.append(data)
                
    return segments