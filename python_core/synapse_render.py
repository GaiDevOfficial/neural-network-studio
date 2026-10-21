"""
Python Synapse Render & Topological Graph Generator
Maps multi-layer neural network parameters (weights, activations, gradients)
to 2D screen coordinates and formats synaptic firing vectors.

File: python_core/synapse_render.py
"""

import math


class SynapseGraphLayout:
    def __init__(self, architecture, width=800, height=500):
        self.architecture = architecture
        self.width = width
        self.height = height
        self.layers = []
        self.synapses = []
        self.compute_layout()

    def compute_layout(self):
        num_layers = len(self.architecture)
        if num_layers == 0:
            return

        pad_x = min(100.0, self.width * 0.12)
        usable_w = self.width - pad_x * 2.0
        layer_spacing = usable_w / (num_layers - 1) if num_layers > 1 else 0.0

        max_display = 14
        self.layers = []

        for l_idx, layer_cfg in enumerate(self.architecture):
            raw_count = layer_cfg["size"]
            is_compressed = raw_count > max_display
            disp_count = max_display if is_compressed else raw_count

            x = pad_x + l_idx * layer_spacing
            node_r = max(8.0, min(18.0, (self.height - 120.0) / (disp_count * 2.5)))
            total_h = (disp_count - 1) * (node_r * 2.5)
            start_y = (self.height - total_h) / 2.0

            nodes = []
            for n in range(disp_count):
                raw_idx = min(raw_count - 1, int((n / (disp_count - 1)) * (raw_count - 1))) if is_compressed else n
                y = start_y + n * (node_r * 2.5)
                nodes.append({
                    "layer": l_idx,
                    "display_index": n,
                    "raw_index": raw_idx,
                    "x": x,
                    "y": y,
                    "radius": node_r
                })
            self.layers.append(nodes)

        # Build synapses
        self.synapses = []
        for l in range(num_layers - 1):
            src_nodes = self.layers[l]
            tgt_nodes = self.layers[l + 1]
            for s in src_nodes:
                for t in tgt_nodes:
                    self.synapses.append({
                        "layer": l,
                        "source": s,
                        "target": t,
                        "weight": 0.0
                    })

    def update_weights(self, weights_snapshot):
        for syn in self.synapses:
            l = syn["layer"]
            s_idx = syn["source"]["raw_index"]
            t_idx = syn["target"]["raw_index"]
            if l < len(weights_snapshot):
                layer_weights = weights_snapshot[l]["weights"]
                if t_idx < len(layer_weights) and s_idx < len(layer_weights[t_idx]):
                    syn["weight"] = float(layer_weights[t_idx][s_idx])

    def generate_svg(self, activations=None, deltas=None):
        """Generates a standalone SVG string of the current network state."""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" style="background:#0a0f1e;">'
        ]

        # Draw Synapses
        for syn in self.synapses:
            w = syn["weight"]
            color = "#00f0ff" if w >= 0 else "#ff007f"
            opacity = max(0.15, min(0.85, 0.2 + abs(w) * 0.4))
            stroke_w = max(0.6, min(5.0, 1.0 + abs(w) * 2.2))
            x1, y1 = syn["source"]["x"], syn["source"]["y"]
            x2, y2 = syn["target"]["x"], syn["target"]["y"]
            svg_parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-opacity="{opacity:.2f}" stroke-width="{stroke_w:.2f}" />'
            )

        # Draw Nodes
        for l_idx, layer in enumerate(self.layers):
            act_layer = activations[l_idx] if activations and l_idx < len(activations) else None
            for node in layer:
                act = float(act_layer[node["raw_index"]]) if act_layer and node["raw_index"] < len(act_layer) else 0.0
                fill = f"rgba(0, 240, 255, {0.2 + act * 0.8:.2f})" if act > 0.05 else "#131b2e"
                svg_parts.append(
                    f'<circle cx="{node["x"]:.1f}" cy="{node["y"]:.1f}" r="{node["radius"]:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="1.5" />'
                )

        svg_parts.append("</svg>")
        return "".join(svg_parts)
