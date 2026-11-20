"""Gallery regressions with synthetic local outputs."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from cstudio import web_gallery as G
from remote.manifest import MAX_SEED


class GalleryTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.jobs = root / "jobs"
        self.jobs.mkdir()
        self.stamp = 1_700_000_001_000_000_000
        with io.BytesIO() as output:
            Image.new("RGBA", (4, 5), (230, 20, 30, 64)).save(output, "PNG")
            self.png_bytes = output.getvalue()
        G._render_thumbnail.cache_clear()
        self.addCleanup(G._render_thumbnail.cache_clear)

    def image(self, folder, name="image", *, stamp=None, data=None):
        png = self.jobs / folder / "out" / f"{name}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(self.png_bytes if data is None else data)
        stamp = self.stamp if stamp is None else stamp
        os.utime(png, ns=(stamp, stamp))
        return png

    def test_invalid_metadata_and_corrupt_image_do_not_hide_valid_items(self):
        bad = self.image("bad-json", stamp=self.stamp + 50)
        (bad.parent.parent / "manifest.json").write_text('{"jobs":', encoding="utf-8")
        (bad.parent / "results.json").write_text('["wrong root type"]', encoding="utf-8")
        edit = self.image("legacy", "edit", stamp=self.stamp + 10)
        self.image("legacy", "unlisted", stamp=self.stamp + 13)
        self.image("legacy", "failed", stamp=self.stamp + 12)
        self.image("corrupt", stamp=self.stamp + 100, data=b"not a PNG")
        self.image("corrupt", "empty", stamp=self.stamp + 101, data=b"")
        self.image("legacy", "contact_sheet", stamp=self.stamp + 11)
        inputs = edit.parent.parent / "inputs"
        inputs.mkdir()
        for name in ("first.png", "second.png"):
            (inputs / name).write_bytes(self.png_bytes)
        (edit.parent.parent / "manifest.json").write_text(json.dumps({
            "_gpu": "G4", "resolution": "2048",
            "jobs": [
                {"id": "edit", "mode": "edit", "edit_resolution": None, "seed": str(MAX_SEED),
                 "inputs": ["inputs/second.png", {}, "../first.png", "inputs/first.png"],
                 "prompt": "test \ud800 preserved", "resolution": ["invalid"]},
                {"id": "failed"}, {"id": "contact_sheet"}, None,
            ],
        }), encoding="utf-8")
        (edit.parent / "results.json").write_text(json.dumps({
            "cu_used": float("nan"), "gpu": "unrecognized GPU",
            "jobs": {"edit": {"status": "done", "seconds": float("inf"), "refined_prompt": ["bad"]},
                     "failed": {"status": "sync_error"}},
        }), encoding="utf-8")
        items, cursor = G.gallery_page(self.jobs, 20)
        self.assertIsNone(cursor)
        self.assertEqual([item["artifact_id"] for item in items], ["bad-json/out/image.png", "legacy/out/edit.png"])
        self.assertEqual(items[0]["prompt"], "")
        recipe = items[1]
        self.assertEqual(recipe["mode"], "edit")
        self.assertEqual(recipe["edit_resolution"], 1024)
        self.assertEqual(recipe["seed"], str(MAX_SEED))
        self.assertEqual(recipe["gpu"], "G4")
        self.assertEqual(recipe["resolution"], "2K")
        self.assertTrue(recipe["has_alpha"])
        self.assertEqual(recipe["prompt"], "test ? preserved")
        self.assertEqual([entry["filename"] for entry in recipe["inputs"]], ["second.png", "first.png"])
        self.assertIsNone(recipe["refined_prompt"])
        self.assertIsNone(recipe["seconds"])
        self.assertIsNone(recipe["cu_used"])
        json.dumps(items, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def test_cursor_uses_tie_breaks_and_nanoseconds_through_gallery_mutation(self):
        self.image("oldest", stamp=self.stamp - 1_000_000)
        for folder in ("old-a", "old-b", "old-c"):
            self.image(folder)
        self.image("new", stamp=self.stamp + 1_000_000)
        first, next_cursor = G.gallery_page(self.jobs, 2)
        self.assertEqual([item["job_dir"] for item in first], ["new", "old-c"])
        self.image("new-after-first", stamp=self.stamp + 2_000_000)
        self.image("old-b").unlink()
        second, cursor = G.gallery_page(self.jobs, 2, G.parse_cursor(next_cursor))
        self.assertEqual([item["job_dir"] for item in second], ["old-a", "oldest"])
        self.assertIsNone(cursor)
        latest, _ = G.gallery_page(self.jobs, 1)
        self.assertEqual(latest[0]["job_dir"], "new-after-first")

    def test_vanishing_candidate_is_skipped_without_reducing_page_size(self):
        keep = self.image("older", stamp=self.stamp - 1)
        gone = self.image("newer")
        candidates = G._candidates

        def after_scan(*args):
            found = candidates(*args)
            gone.unlink()
            return found

        with patch.object(G, "_candidates", side_effect=after_scan):
            items, cursor = G.gallery_page(self.jobs, 1)
        self.assertEqual(items[0]["filename"], keep.name)
        self.assertEqual(items[0]["job_dir"], "older")
        self.assertIsNone(cursor)

    def test_large_gallery_restricts_metadata_and_header_reads_to_the_page(self):
        for folder in range(40):
            for number in range(30):
                self.image(f"batch-{folder:03}", f"image-{number:03}")
        with (
            patch.object(G, "read_gallery_metadata", wraps=G.read_gallery_metadata) as read,
            patch.object(G, "inspect_image_meta", wraps=G.inspect_image_meta) as header,
        ):
            items, cursor = G.gallery_page(self.jobs, 12)
        self.assertEqual([item["job_id"] for item in items], [f"image-{i:03}" for i in range(29, 17, -1)])
        self.assertEqual({item["job_dir"] for item in items}, {"batch-039"})
        self.assertEqual(read.call_count, 2)
        self.assertEqual(header.call_count, 13)
        older, _ = G.gallery_page(self.jobs, 12, G.parse_cursor(cursor))
        self.assertEqual(older[0]["job_id"], "image-017")

    def test_thumbnail_preserves_alpha_and_invalidates_cached_replacement(self):
        png = self.image("batch")
        Image.new("RGBA", (384, 192), (230, 20, 30, 64)).save(png, compress_level=0)
        first_stat = png.stat()
        with patch.object(Image, "open", wraps=Image.open) as opened:
            first = G.gallery_thumbnail(png)
            self.assertEqual(G.gallery_thumbnail(png), first)
            opened.assert_called_once()
            replacement = png.with_suffix(".staged")
            Image.new("RGBA", (384, 192), (20, 230, 30, 96)).save(replacement, format="PNG", compress_level=0)
            os.utime(replacement, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
            self.assertEqual(replacement.stat().st_size, first_stat.st_size)
            replacement.replace(png)
            second = G.gallery_thumbnail(png)
            self.assertEqual(opened.call_count, 2)
        self.assertNotEqual(first, second)
        with Image.open(io.BytesIO(second)) as image:
            self.assertEqual(image.size, (192, 96))
            self.assertEqual(image.getpixel((0, 0))[3], 96)
        with Image.open(png) as image:
            self.assertEqual(image.size, (384, 192))

    def test_thumbnail_dimension_limit_preserves_original_file_access(self):
        png = self.image("batch")
        with patch.object(G, "THUMBNAIL_MAX_PIXELS", 15), self.assertRaises(HTTPException) as exc:
            G.gallery_thumbnail(png)
        self.assertEqual(exc.exception.status_code, 404)
        self.assertEqual(G.safe_subfile(self.jobs, "batch", "out", "image.png").read_bytes(), self.png_bytes)
