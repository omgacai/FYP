# Raster floor-plan image → graph: literature position

## Bottom line

Do **not** claim that no floor-plan image-to-graph conversion exists. It does.
The defensible gap is that room-instance, adjacency, and especially door/access
connectivity remain difficult and method/dataset dependent; few reusable pipelines
directly match CubiCasa5K source SVG labels and this project's two-predicate graph.

| Work | Year | Image input → output | FYP use |
| --- | --- | --- | --- |
| [Chen & Stouffs, Robust Attributed Adjacency Graph Extraction](https://github.com/JanineCHEN/AAG-FP) | 2022 | Raster floor plan → attributed adjacency graph | Closest direct, open-code baseline candidate; check its graph semantics before comparison. |
| [Yang et al., Automated Semantics and Topology Representation](https://doi.org/10.1109/JSTARS.2022.3205746) | 2022 | Raster → learned primitives → constrained planar/topological graph | Shows topology needs structured reconstruction, not segmentation masks alone. |
| [Knechtel et al., Semantic Floorplan Segmentation using Self-Constructing Graph Networks](https://doi.org/10.1016/j.autcon.2024.105649) | 2024 | Raster → multi-task segmentation with graph reasoning | Recent parser alternative; supports later layout-graph inference but is not a drop-in CubiCasa room-door benchmark. |
| [Wen et al., Floor Plan Restoration: A Multimodal Method Under One Second](https://doi.org/10.1109/TVCG.2025.3539497) | 2025 | Raster → semantic/vector restoration using names, icons, and boundaries | Recent evidence that multimodal floor-plan cues help parsing; graph topology remains downstream. |

Your CNN → pseudo-SVG → CubiGraph route is a modular baseline, not the only method
or a claim of novelty. Evaluate it against source-SVG-derived labels. If time permits,
compare with AAG-FP; the stronger contribution is a reproducible CubiCasa graph audit
and evidence of downstream retrieval value.
