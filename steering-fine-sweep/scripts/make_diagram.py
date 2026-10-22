import json
import os
import subprocess

def make_diagram():
    elements = []
    
    # Title
    elements.append({
        "id": "title",
        "type": "text",
        "x": 60,
        "y": 30,
        "width": 1100,
        "height": 30,
        "angle": 0,
        "strokeColor": "#1e293b",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "roundness": None,
        "seed": 1001,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "text": "Finer Steering Sweep Pipeline & Decision Rule",
        "fontSize": 24,
        "fontFamily": 1,
        "textAlign": "left",
        "verticalAlign": "top"
    })

    boxes = [
        {
            "id": "box1",
            "x": 60,
            "y": 80,
            "w": 360,
            "h": 240,
            "bg": "#e0f2fe",
            "stroke": "#0369a1",
            "title": "Stage 1: Coarse Pilot Grid",
            "lines": [
                "10 cells: L in {12, 20, 28} x s in {1.5, 2.5, 3.5}",
                "20 pilot prompts + 50 pilot MMLU items",
                "Observed: L12 collapses (30% MMLU at s=2.5)",
                "L20/L28 delay collapse but relevance falls",
                "L24 identified as candidate sweet spot"
            ]
        },
        {
            "id": "box2",
            "x": 460,
            "y": 80,
            "w": 360,
            "h": 240,
            "bg": "#fef3c7",
            "stroke": "#b45309",
            "title": "Pre-Registration Lock (Git 17da4ce)",
            "lines": [
                "Candidate Cell: Layer 24, Strength s = 2.8",
                "Quantitative Pre-Registered Targets:",
                "1. Topic Adherence >= 70% (CI lower >= 60%)",
                "2. MMLU Drop <= 3.0 pp (CI upper <= 5.0 pp)",
                "3. Useful Coherence >= 60% (CI lower >= 50%)",
                "Committed BEFORE Confirmatory Run"
            ]
        },
        {
            "id": "box3",
            "x": 860,
            "y": 80,
            "w": 360,
            "h": 240,
            "bg": "#dcfce7",
            "stroke": "#15803d",
            "title": "Stage 2: Confirmatory Grid",
            "lines": [
                "36 cells: 5 layers {12, 16, 20, 24, 28}",
                "x 7 strengths {1.5, 2.0, 2.5, 2.8, 3.1, 3.5, 4.0}",
                "+ unsteered baseline (L=0, s=0)",
                "100 FRESH Dolly prompts (open QA)",
                "250 FRESH MMLU test items (knowledge)"
            ]
        },
        {
            "id": "box4",
            "x": 60,
            "y": 360,
            "w": 560,
            "h": 230,
            "bg": "#f3e8ff",
            "stroke": "#6b21a8",
            "title": "Dual-Axis Evaluation per Cell",
            "lines": [
                "1. Steered Generation (100 prompts) & MMLU (250 items)",
                "2. Unsteered Mechanical Fluency:",
                "   • PPL under unsteered Qwen3-4B <= baseline P95",
                "   • Repeated trigram fraction <= 0.20",
                "3. Unsteered Qwen3-4B Judge:",
                "   • Topic: Mentions Eiffel/Paris (regex + judge > 0.5)",
                "   • Useful Coherence = Fluent AND Relevant to Prompt"
            ]
        },
        {
            "id": "box5",
            "x": 660,
            "y": 360,
            "w": 560,
            "h": 230,
            "bg": "#ffe4e6",
            "stroke": "#be123c",
            "title": "Pre-Registered Decision Protocol",
            "lines": [
                "PROVEN: Candidate cell (L24, s2.8) or ANY cell clears:",
                "Topic >= 70%, MMLU drop <= 3.0 pp, Useful >= 60%",
                "DISPROVEN: Across all 35 cells, no cell meets the rule;",
                "topic steering strictly trades off against answering",
                "INCONCLUSIVE: Margin overlaps or unresolvable CI bounds"
            ]
        }
    ]

    for b in boxes:
        # Rectangle
        elements.append({
            "id": f"{b['id']}_rect",
            "type": "rectangle",
            "x": b["x"],
            "y": b["y"],
            "width": b["w"],
            "height": b["h"],
            "angle": 0,
            "strokeColor": b["stroke"],
            "backgroundColor": b["bg"],
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "roundness": {"type": 3},
            "seed": 2000 + len(elements),
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False
        })
        # Title text
        elements.append({
            "id": f"{b['id']}_title",
            "type": "text",
            "x": b["x"] + 14,
            "y": b["y"] + 12,
            "width": b["w"] - 28,
            "height": 24,
            "angle": 0,
            "strokeColor": b["stroke"],
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "roundness": None,
            "seed": 3000 + len(elements),
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "text": b["title"],
            "fontSize": 17,
            "fontFamily": 1,
            "textAlign": "left",
            "verticalAlign": "top"
        })
        # Individual text lines
        y_cursor = b["y"] + 46
        for line_idx, line in enumerate(b["lines"]):
            elements.append({
                "id": f"{b['id']}_line_{line_idx}",
                "type": "text",
                "x": b["x"] + 14,
                "y": y_cursor,
                "width": b["w"] - 28,
                "height": 20,
                "angle": 0,
                "strokeColor": "#1e293b",
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": "solid",
                "roughness": 1,
                "opacity": 100,
                "groupIds": [],
                "roundness": None,
                "seed": 4000 + len(elements),
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "text": line,
                "fontSize": 14,
                "fontFamily": 1,
                "textAlign": "left",
                "verticalAlign": "top"
            })
            y_cursor += 26

    def make_arrow(aid, x1, y1, x2, y2, color="#64748b"):
        return {
            "id": aid,
            "type": "arrow",
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "roundness": {"type": 2},
            "seed": 5000 + len(elements),
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "points": [[0, 0], [x2 - x1, y2 - y1]],
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow"
        }

    # Box1 -> Box2
    elements.append(make_arrow("arr1", 420, 200, 460, 200))
    # Box2 -> Box3
    elements.append(make_arrow("arr2", 820, 200, 860, 200))
    # Box3 down to Box5
    elements.append(make_arrow("arr3", 1040, 320, 1040, 360))
    # Box4 -> Box5
    elements.append(make_arrow("arr4", 620, 475, 660, 475))

    data = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff"
        },
        "files": {}
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "diagram.excalidraw")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote", out_path)

if __name__ == "__main__":
    make_diagram()
