import json
from pathlib import Path
from typing import List, Dict

def load_tickets(path: str = "../sample_tickets.json") -> List[Dict]:
    p = Path(__file__).parent.joinpath(path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    return tickets

if __name__ == "__main__":
    tickets = load_tickets()
    print(f"Loaded {len(tickets)} tickets")
    # show first ticket
    import pprint
    pprint.pprint(tickets[0])
