import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from initial_audit import verify_initial_pair


class InitialObservationChecks(unittest.TestCase):
    def pair(self, root, changed=1, delta=1, physical_difference=False):
        paths=[]
        for arm in range(2):
            p=Path(root)/str(arm);p.mkdir()
            image=np.zeros((32,32,3),dtype=np.uint8)
            if arm:image.reshape(-1,3)[:changed,0]=delta
            Image.fromarray(image).save(p/'observation_000.png')
            (p/'initial.json').write_text(json.dumps({'qpos':[int(physical_difference and arm==1)],
                'qvel':[0], 'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest()}))
            paths.append(p)
        return paths

    def test_explicit_roundoff_mode_accepts_one_level_but_strict_mode_rejects_it(self):
        with tempfile.TemporaryDirectory() as root:
            pair=self.pair(root)
            self.assertEqual(verify_initial_pair(*pair,max_changed_pixels=8)['changed_pixels'],1)
            with self.assertRaisesRegex(RuntimeError,'renderer-roundoff'):
                verify_initial_pair(*pair)

    def test_ninth_changed_pixel_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'renderer-roundoff'):
                verify_initial_pair(*self.pair(root,changed=9),max_changed_pixels=8)

    def test_two_level_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'renderer-roundoff'):
                verify_initial_pair(*self.pair(root,delta=2),max_changed_pixels=8)

    def test_declared_new_attempt_bound_accepts_roundoff_but_rejects_larger_changes(self):
        from initial_audit import PROVISIONAL_CHANGED_PIXELS
        self.assertEqual(PROVISIONAL_CHANGED_PIXELS, 26)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(verify_initial_pair(*self.pair(root,changed=13),
                max_changed_pixels=PROVISIONAL_CHANGED_PIXELS)['changed_pixels'],13)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'renderer-roundoff'):
                verify_initial_pair(*self.pair(root,changed=27),max_changed_pixels=PROVISIONAL_CHANGED_PIXELS)

    def test_physical_differences_are_never_tolerated(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'physical state mismatch'):
                verify_initial_pair(*self.pair(root,changed=0,physical_difference=True),max_changed_pixels=8)

    def test_saved_image_must_match_the_native_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            pair=self.pair(root,changed=0)
            state=json.loads((pair[0]/'initial.json').read_text());state['rgb_sha256']='forged'
            (pair[0]/'initial.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError,'fingerprint'):
                verify_initial_pair(*pair,max_changed_pixels=8)
