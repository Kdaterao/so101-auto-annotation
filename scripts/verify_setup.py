#!/usr/bin/env python3
"""Check installed CLI imports, a real CUDA operation, and CPU video decoding.

No model weights, datasets, authentication, or inference jobs are downloaded.
Run through linux_setup.sh --profile annotation --check in the target venv.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def check_package():
    import robot101
    expected = ROOT / "src/robot101/__init__.py"
    if Path(robot101.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"robot101 imports from {robot101.__file__}, expected {expected}")
    print("Installed package:", robot101.__file__, flush=True)


def check_video(directory):
    import av
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    path = Path(directory) / "setup_decode.mp4"
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=5)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for value in (0, 80, 160):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    decoder = VideoDecoder(str(path), device="cpu")
    if len(decoder) != 3 or tuple(decoder[0].data.shape) != (3, 32, 32):
        raise RuntimeError("TorchCodec failed the three-frame CPU decode check")
    print("Video check: PyAV encode and TorchCodec CPU decode passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["annotation", "grounded", "robot", "molmo2"], required=True)
    args = parser.parse_args()
    marker = Path(sys.prefix) / ".setup-ready.json"
    marker.unlink(missing_ok=True)
    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run verification with the profile's Python 3.12 virtual environment")
    check_package()

    import torch
    import torchvision
    if torch.__version__.split("+")[0] != "2.10.0" or torchvision.__version__.split("+")[0] != "0.25.0":
        raise RuntimeError("Expected PyTorch 2.10.0 / torchvision 0.25.0; rerun setup to repair versions")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and PyTorch CUDA wheel")
    result = torch.ones((4, 4), device="cuda") @ torch.ones((4, 4), device="cuda")
    torch.cuda.synchronize()
    if result[0, 0].item() != 4:
        raise RuntimeError("CUDA matrix multiplication returned the wrong result")
    print("CUDA check:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)

    modules = ["datasets", "huggingface_hub", "av", "torchcodec"]
    if args.profile == "annotation":
        modules += ["anthropic", "timm", "cv2", "psutil", "einops", "accelerate"]
        from transformers import PerceptionLMForConditionalGeneration, Qwen3VLForConditionalGeneration
        from robot101.perception.annotation_models import JEPAEncoder, Captioner
    elif args.profile == "grounded":
        modules += ["spacy", "einshape", "einops", "tree", "tapnet.torch.tapir_model",
                    "sam2.sam2_video_predictor"]
        from transformers import AutoModelForImageTextToText
        from robot101.perception.molmo2 import Molmo2Worker
        from robot101.perception.tracking import BootsTAPIR
        import spacy
        spacy.load("en_core_web_sm")
    elif args.profile == "molmo2":
        modules += ["omegaconf", "olmo"]
    for name in modules:
        importlib.import_module(name)

    commands = (["robot101.data.annotate", "robot101.data.aggregate_annotations"]
                if args.profile == "annotation" else
                ["robot101.data.preprocess"] if args.profile == "grounded" else [])
    if args.profile in {"annotation", "grounded", "robot"}:
        from robot101.data.utilities.episode_helpers import _load_lerobot
        dataset, _, _, _ = _load_lerobot()
        import inspect
        if not Path(inspect.getfile(dataset)).resolve().is_relative_to((ROOT / "lerobot").resolve()):
            raise RuntimeError("LeRobot resolved to another checkout rather than this repo's pinned submodule")

    with tempfile.TemporaryDirectory(prefix="so101-setup-check-") as directory:
        # No cwd or environment path may make an otherwise broken install pass.
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        for module in commands:
            subprocess.run([sys.executable, "-I", "-m", module, "--help"], cwd=directory,
                           env=env, stdout=subprocess.DEVNULL, check=True)
            print("CLI check:", module, flush=True)
        check_video(directory)

    versions = {name: importlib.metadata.version(name)
                for name in ("so101-auto-annotation", "torch", "torchvision", "torchcodec", "transformers")}
    report = dict(profile=args.profile, python=sys.executable, package_root=str(ROOT), versions=versions,
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(marker)
    print("Environment checks passed:", marker, flush=True)


if __name__ == "__main__":
    main()
