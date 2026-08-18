from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from floorplan_app.core.models import ParserResult
from floorplan_app.core.registry import register_parser
from floorplan_app.core.parser_base import FloorplanParser


@register_parser('CubiCasa5K (pretrained)')
class CubiCasa5KParser(FloorplanParser):
    """Adapter around the original CubiCasa5K FloorTrans checkpoint."""

    display_name = 'CubiCasa5K (pretrained)'
    description = 'FloorTrans hourglass model; predicts rooms/walls, icons, and structural heatmaps.'

    room_class_names = [
        'Background', 'Outdoor', 'Wall', 'Kitchen', 'Living Room', 'Bed Room',
        'Bath', 'Entry', 'Railing', 'Storage', 'Garage', 'Undefined',
    ]
    icon_class_names = [
        'No Icon', 'Window', 'Door', 'Closet', 'Electrical Appliance', 'Toilet',
        'Sink', 'Sauna Bench', 'Fire Place', 'Bathtub', 'Chimney',
    ]

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.repo = self.project_root / 'third_party' / 'CubiCasa5k'
        self.checkpoint = self.repo / 'model_best_val_loss_var.pkl'

    def _load_model(self):
        import torch

        if not self.repo.exists() or not self.checkpoint.exists():
            raise FileNotFoundError(
                'CubiCasa5K repository/checkpoint missing. Run the setup cell in '
                'cubicasa5k_cubigraph_macos.ipynb first.'
            )
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from floortrans.models import get_model

        # Original CubiCasa code uses a relative path for model_1427.pth.
        previous_cwd = Path.cwd()
        os.chdir(self.repo)
        try:
            model = get_model('hg_furukawa_original', 51)
        finally:
            os.chdir(previous_cwd)
        model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        return model.eval()

    @staticmethod
    def _patch_scipy_mode() -> None:
        import scipy.stats as stats
        import floortrans.post_prosessing

        original_mode = stats.mode

        def patched_mode(*args, **kwargs):
            result = original_mode(*args, **kwargs)
            if hasattr(result, 'mode') and np.isscalar(result.mode):
                class LegacyMode:
                    def __init__(self, value):
                        self.mode = [value]
                return LegacyMode(result.mode)
            return result

        floortrans.post_prosessing.stats.mode = patched_mode

    @staticmethod
    def _mean_class_probability(geometry, class_probability: np.ndarray) -> float | None:
        """Mean probability of this geometry's assigned class over its interior pixels."""
        if geometry.is_empty or geometry.area <= 1e-6:
            return None
        from matplotlib.path import Path as MatplotlibPath

        min_x, min_y, max_x, max_y = geometry.bounds
        x0, y0 = max(0, int(np.floor(min_x))), max(0, int(np.floor(min_y)))
        x1, y1 = min(class_probability.shape[1], int(np.ceil(max_x)) + 1), min(class_probability.shape[0], int(np.ceil(max_y)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside = MatplotlibPath(np.asarray(geometry.exterior.coords)).contains_points(points)
        for interior in geometry.interiors:
            inside &= ~MatplotlibPath(np.asarray(interior.coords)).contains_points(points)
        values = class_probability[y0:y1, x0:x1].ravel()[inside]
        return float(values.mean()) if values.size else None

    def parse(self, image_path: Path, *, max_side: int = 1024, postprocess_threshold: float = 0.2) -> ParserResult:
        import torch
        import torch.nn.functional as F

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from floortrans.loaders import RotateNTurns
        from floortrans.plotting import polygons_to_image
        from floortrans.post_prosessing import get_polygons, split_prediction

        self._patch_scipy_mode()
        source = Image.open(image_path).convert('RGB')
        scale = min(1.0, max_side / max(source.size))
        resized = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
        padded_w, padded_h = ((resized.width + 31) // 32 * 32, (resized.height + 31) // 32 * 32)
        canvas = Image.new('RGB', (padded_w, padded_h), 'white')
        canvas.paste(resized, (0, 0))
        image_array = (np.asarray(canvas).astype(np.float32) / 255.0 - 0.5) * 2.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = self._load_model().to(device)
        rot = RotateNTurns()

        def infer(target):
            with torch.no_grad():
                predictions = []
                for forward, backward in [(0, 0), (1, -1), (2, 2), (-1, 1)]:
                    predicted = model(rot(tensor.to(target), 'tensor', forward))
                    predicted = rot(predicted, 'tensor', backward)
                    predicted = rot(predicted, 'points', backward)
                    predictions.append(F.interpolate(predicted, size=(padded_h, padded_w), mode='bilinear', align_corners=True))
                return torch.stack(predictions).mean(0).cpu()

        try:
            prediction = infer(device)
        except RuntimeError as error:
            if device.type != 'mps':
                raise
            device = torch.device('cpu')
            model = model.to(device)
            prediction = infer(device)

        heatmaps, room_logits, icon_logits = split_prediction(prediction, (padded_h, padded_w), [21, 12, 11])
        polygons, types, room_polygons, room_types = get_polygons(
            (heatmaps, room_logits, icon_logits), postprocess_threshold, [1, 2]
        )
        # CubiCasa's split_prediction already applies softmax and returns NumPy
        # probability maps, one channel per semantic class. Do not softmax again.
        room_probs = room_logits
        icon_probs = icon_logits
        # Split disconnected regions now. This gives the dashboard one row and
        # one confidence value per graph candidate rather than per semantic union.
        room_components, room_component_types = [], []
        for geometry, metadata in zip(room_polygons, room_types):
            components = list(geometry.geoms) if hasattr(geometry, 'geoms') else [geometry]
            for component in components:
                if component.is_empty or component.area <= 1e-6:
                    continue
                component_metadata = dict(metadata)
                component_metadata['mean_class_confidence'] = self._mean_class_probability(
                    component, room_probs[component_metadata['class']]
                )
                room_components.append(component)
                room_component_types.append(component_metadata)
        room_seg, icon_seg = polygons_to_image(
            polygons, types, room_components, room_component_types, padded_h, padded_w
        )
        room_best = room_probs.max(axis=0)
        icon_best = icon_probs.max(axis=0)

        return ParserResult(
            parser_name=self.display_name,
            source_image=source,
            processed_image=canvas,
            room_segmentation=room_seg,
            icon_segmentation=icon_seg,
            room_polygons=room_components,
            room_metadata=room_component_types,
            icon_polygons=list(polygons),
            icon_metadata=list(types),
            room_class_names=self.room_class_names,
            icon_class_names=self.icon_class_names,
            confidence_maps={'room_best': room_best, 'icon_best': icon_best},
            diagnostics={
                'device': str(device), 'input_size': source.size, 'processed_size': canvas.size,
                'room_pixel_confidence_mean': float(room_best.mean()),
                'room_pixel_confidence_p10': float(np.quantile(room_best, 0.10)),
                'icon_pixel_confidence_mean': float(icon_best.mean()),
                'postprocess_threshold': postprocess_threshold,
            },
        )
