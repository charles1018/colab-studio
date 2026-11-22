"""Offline checks for the pinned ComfyUI Edit reference graph and image contract."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageOps

from cstudio import runner
from remote import remote_runner


QWEN = runner.get_model_plugin()


def evaluate_reference(graph, link, input_dir):
    """Use Pillow stand-ins for the pinned image nodes involved in Edit preprocessing."""
    node_id, slot = link
    node = graph[node_id]
    inputs = {
        key: evaluate_reference(graph, value, input_dir) if isinstance(value, list) else value
        for key, value in node["inputs"].items()
    }
    kind = node["class_type"]
    if kind == "LoadImage":
        with Image.open(input_dir / inputs["image"]) as image:
            mask = ImageOps.invert(image.getchannel("A")) if "A" in image.getbands() else Image.new("L", (64, 64))
            return (image.convert("RGB"), mask)[slot]
    if slot != 0:
        raise AssertionError(f"unexpected {kind} output {slot}")
    if kind == "JoinImageWithAlpha":
        image = inputs["image"].copy()
        image.putalpha(ImageOps.invert(inputs["alpha"].resize(image.size, Image.Resampling.BILINEAR)))
        return image
    if kind == "EmptyImage":
        color = inputs["color"].to_bytes(3, "big")
        return Image.new("RGB", (inputs["width"], inputs["height"]), tuple(color))
    if kind == "ImageCompositeMasked":
        dest = inputs["destination"]
        if (inputs["x"], inputs["y"], inputs["resize_source"]) != (0, 0, True):
            raise AssertionError("expected a resized source over the entire reference")
        mask = inputs["mask"].resize(dest.size, Image.Resampling.BILINEAR)
        return Image.composite(inputs["source"].resize(dest.size), dest, mask)
    raise AssertionError(f"unexpected reference node: {kind}")


class EditAlphaTests(unittest.TestCase):
    def test_pe_sends_textgenerate_dynamic_combo_with_dotted_sampling_fields(self):
        for mode, presence_penalty in (("t2i", 1.5), ("edit", 0.0)):
            with self.subTest(mode=mode):
                job = {"id": mode, "mode": mode, "prompt": "Generate", "seed": 41, "refine_prompt": True}
                if mode == "edit":
                    job["inputs"] = ["ref.png"]
                graph = QWEN.build_qwen21_graph(job)
                textgen = graph["32"]
                self.assertEqual(textgen["class_type"], "TextGenerate")
                self.assertEqual({key: value for key, value in textgen["inputs"].items()
                                  if key.split(".", 1)[0] == "sampling_mode"}, {
                    "sampling_mode": "on",
                    "sampling_mode.temperature": 1.0,
                    "sampling_mode.top_k": 20,
                    "sampling_mode.top_p": 0.95,
                    "sampling_mode.min_p": 0.0,
                    "sampling_mode.repetition_penalty": 1.0,
                    "sampling_mode.presence_penalty": presence_penalty,
                    "sampling_mode.seed": 41,
                })

    def test_staged_png_preserves_alpha_in_edit_and_flattens_white_for_pe(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            inputs = job_dir / "inputs"
            inputs.mkdir()
            comfy = job_dir / "ComfyUI"
            comfy.mkdir()
            color = (23, 60, 90)
            rgba_pixels = [(15, 30, 60, 255), (220, 10, 30, 0), (10, 110, 210, 128)]
            rgba = Image.new("RGBA", (3, 1))
            rgba.putdata(rgba_pixels)
            rgba.save(inputs / "rgba.png")
            Image.new("RGB", (1, 2), color).save(inputs / "rgb.png")
            names = ["rgba.png", "rgb.png"]
            job = {"id": "Edit", "mode": "edit", "prompt": "Move these", "inputs": names, "refine_prompt": True}
            with patch.multiple(remote_runner, JOB=job_dir, COMFY=comfy):
                staged = [remote_runner.stage_input_image(name, job["id"], i) for i, name in enumerate(names, 1)]
            graph = QWEN.build_qwen21_graph(job, uploaded_inputs=staged)
            encode = graph["40"]["inputs"]
            edit_images = [evaluate_reference(graph, encode[f"images.image_{i}"], comfy / "input") for i in (1, 2)]
            pe_images = [evaluate_reference(graph, link, comfy / "input") for link in graph["31"]["inputs"].values()]

            self.assertEqual([image.size for image in edit_images], [(3, 1), (1, 2)])
            self.assertEqual([image.size for image in pe_images], [(3, 1), (1, 2)])
            self.assertEqual([image.mode for image in edit_images], ["RGBA", "RGBA"])
            self.assertEqual([image.mode for image in pe_images], ["RGB", "RGB"])
            self.assertEqual([edit_images[0].getpixel((x, 0)) for x in range(3)], rgba_pixels)
            self.assertEqual([pe_images[0].getpixel((x, 0)) for x in range(3)], [rgba_pixels[0][:3], (255, 255, 255), (132, 182, 232)])
            self.assertEqual(edit_images[1].getpixel((0, 1)), (*color, 255))
            self.assertEqual(pe_images[1].getpixel((0, 1)), color)

    def test_one_through_ten_references_keep_order_in_both_edit_paths(self):
        for count in range(1, 11):
            names = [f"{index}.png" for index in reversed(range(count))]
            for refine in (False, True):
                with self.subTest(count=count, refine=refine):
                    graph = QWEN.build_qwen21_graph({
                        "id": "edit", "mode": "edit", "prompt": "Edit these", "inputs": names, "refine_prompt": refine,
                    })
                    encode = graph["40"]["inputs"]
                    self.assertEqual(len([key for key in encode if key.startswith("images.")]), count)
                    for i, name in enumerate(names):
                        source = str(20 + i)
                        rgba = str(70 + i)
                        self.assertEqual(graph[source], {"class_type": "LoadImage", "inputs": {"image": name}})
                        self.assertEqual(graph[rgba], {
                            "class_type": "JoinImageWithAlpha", "inputs": {"image": [source, 0], "alpha": [source, 1]},
                        })
                        self.assertEqual(encode[f"images.image_{i + 1}"], [rgba, 0])
                    if refine:
                        self.assertEqual(graph["31"]["class_type"], "BatchImagesNode")
                        self.assertEqual(graph["32"]["inputs"]["image"], ["31", 0])
                        slots = graph["31"]["inputs"].items()
                        self.assertEqual(len(slots), count)
                        self.assertEqual(list(slots), sorted(slots, key=lambda item: int(item[0].removeprefix("images.image"))))
                        for i, (_, [node_id, slot]) in enumerate(slots):
                            self.assertEqual(slot, 0)
                            self.assertEqual(graph[node_id], {
                                "class_type": "ImageCompositeMasked",
                                "inputs": {"destination": [str(20 + i), 0], "source": ["90", 0],
                                           "mask": [str(20 + i), 1], "x": 0, "y": 0, "resize_source": True},
                            })
                    else:
                        self.assertNotIn("31", graph)
                        self.assertNotIn("90", graph)
                        self.assertFalse(any(node["class_type"] == "ImageCompositeMasked" for node in graph.values()))

    def test_edit_keeps_reference_resolution_and_latent_size_choices(self):
        base = {"id": "edit", "mode": "edit", "prompt": "Edit it", "inputs": ["ref.png"]}
        for settings, resolution in (({}, 1024), ({"resolution": "2K"}, 2048), ({"edit_resolution": 0}, 0)):
            with self.subTest(settings=settings):
                graph = QWEN.build_qwen21_graph({**base, **settings})
                self.assertEqual(graph["40"]["inputs"]["resolution"], resolution)
                self.assertEqual(graph["54"]["inputs"]["latent_image"], ["40", 2])
        graph = QWEN.build_qwen21_graph({**base, "edit_resolution": 2048, "width": 512, "height": 768})
        self.assertEqual(graph["40"]["inputs"]["resolution"], 2048)
        self.assertEqual(graph["54"]["inputs"]["latent_image"], ["41", 0])
        self.assertEqual(graph["41"]["inputs"], {"width": 512, "height": 768, "batch_size": 1})

    def test_pre_alpha_edit_result_is_stale_and_t2i_has_no_edit_nodes(self):
        edit = {"id": "edit", "mode": "edit", "prompt": "Edit", "inputs": ["ref.png"], "refine_prompt": True}
        t2i = {"id": "t2i", "prompt": "Generate", "transparent": True, "refine_prompt": True}
        manifest = {"jobs": [edit, t2i]}
        graph = QWEN.build_qwen21_graph(edit, manifest=manifest)
        legacy_edit = deepcopy(graph)
        for node_id in ("70", "80", "90"):
            del legacy_edit[node_id]
        legacy_edit["40"]["inputs"]["images.image_1"] = ["20", 0]
        legacy_edit["31"]["inputs"]["images.image0"] = ["20", 0]
        t2i_graph = QWEN.build_qwen21_graph(t2i, manifest=manifest)
        self.assertFalse({"LoadImage", "JoinImageWithAlpha", "ImageCompositeMasked", "BatchImagesNode"} & {
            node["class_type"] for node in t2i_graph.values()
        })
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.new("RGBA", (1, 1), (200, 40, 50, 128)).save(base / "ref.png")
            (base / "out").mkdir()
            pins = runner.load_versions()
            with patch.object(QWEN, "build_qwen21_graph", side_effect=[legacy_edit, t2i_graph]):
                records = {
                    job["id"]: {"fingerprint": runner.compute_job_fingerprint(job, manifest, base, versions=pins)}
                    for job in manifest["jobs"]
                }
            for job in manifest["jobs"]:
                jid = job["id"]
                image = base / "out" / f"{jid}.png"
                Image.new("RGB", (1, 1)).save(image)
                records[jid].update(status="done", file=f"out/{jid}.png", bytes=image.stat().st_size,
                                    sha256=runner.file_sha256(image))
            (base / "out" / "results.json").write_text(json.dumps({"jobs": records}), encoding="utf-8")
            _, todo, done, stale = runner.classify_jobs(manifest, base)
            self.assertEqual(todo, [edit])
            self.assertEqual(done, ["t2i"])
            self.assertEqual([job_id for job_id, _ in stale], ["edit"])

    def test_request_fingerprint_changes_for_effective_graph_and_ordered_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.new("RGBA", (2, 2), (250, 0, 0, 50)).save(base / "a.png")
            Image.new("RGBA", (2, 2), (0, 250, 0, 200)).save(base / "b.png")
            edit = {"id": "edit", "mode": "edit", "prompt": "Edit", "inputs": ["a.png", "b.png"],
                    "refine_prompt": True, "custom_size": True}
            t2i = {"id": "t2i", "prompt": "Generate"}

            def fingerprints(edit_job=edit, t2i_job=t2i, **options):
                return runner.classify_jobs({"jobs": [edit_job, t2i_job]}, base, **options)[0]

            original = fingerprints()
            for changes in (
                {"inputs": ["b.png", "a.png"]}, {"seed": 17}, {"resolution": "2K"}, {"pe_variant": "w4a8"},
                {"refine_prompt": False}, {"quant": "BF16"}, {"text_encoder_variant": "w4a8"},
            ):
                with self.subTest(changes=changes):
                    after = fingerprints(edit_job=dict(edit, **changes))
                    self.assertNotEqual(after["edit"], original["edit"])
                    self.assertEqual(after["t2i"], original["t2i"])
            self.assertNotEqual(fingerprints(t2i_job=dict(t2i, transparent=True))["t2i"], original["t2i"])
            self.assertNotEqual(fingerprints(cli_quant="Q4_K_M")["edit"], original["edit"])
            Image.new("RGBA", (2, 2), (250, 0, 0, 51)).save(base / "a.png")
            self.assertNotEqual(fingerprints()["edit"], original["edit"])
