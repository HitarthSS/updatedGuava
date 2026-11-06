"""Exact physical starts and explicitly bounded native-renderer roundoff."""
import hashlib
import json
from pathlib import Path

# Declared before the new evaluation: <= 0.01% of 512x512 pixels, each
# differing by at most one intensity level. Strict mode remains exact.
PROVISIONAL_CHANGED_PIXELS = 26


def verify_initial_pair(first, second, max_changed_pixels=0):
    import numpy as np
    from PIL import Image

    directories = [Path(first), Path(second)]
    records = [json.loads((directory/'initial.json').read_text()) for directory in directories]
    physical = [{key: value for key,value in record.items() if key != 'rgb_sha256'} for record in records]
    if physical[0] != physical[1]:
        raise RuntimeError('Initial physical state mismatch across arms')
    images = [np.asarray(Image.open(directory/'observation_000.png').convert('RGB')) for directory in directories]
    if images[0].shape != images[1].shape:
        raise RuntimeError('Initial image dimensions differ across arms')
    for record, image in zip(records, images):
        if hashlib.sha256(image.tobytes()).hexdigest() != record['rgb_sha256']:
            raise RuntimeError('Initial image fingerprint does not match saved native observation')
    difference = np.abs(images[0].astype(np.int16)-images[1].astype(np.int16))
    changed = int(difference.any(axis=2).sum())
    maximum = int(difference.max())
    if changed > max_changed_pixels or maximum > 1:
        raise RuntimeError('Initial image mismatch beyond declared renderer-roundoff limit')
    return {'physical_state_exact': True, 'rgb_exact': changed == 0,
            'changed_pixels': changed, 'total_pixels': int(images[0].shape[0]*images[0].shape[1]),
            'maximum_rgb_channel_delta': maximum, 'max_allowed_changed_pixels': max_changed_pixels,
            'max_allowed_rgb_channel_delta': 1 if max_changed_pixels else 0,
            'images_modified': False}
