"""Train the 3DGS model for a scene with gsplat v1.4.0's unmodified examples/simple_trainer.py.

Same CLI as simple_trainer.py. Two differences from running it directly:
  * the training views are exactly split.json's train_frames / training_frames. Stock gsplat
    holds out every 8th image, and would also train on the close-ups, which are in images/
    and in the COLMAP model;
  * world-space normalization is off, so the checkpoint is in the raw COLMAP frame that
    src/evaluate.py renders the close-ups in.

Usage:
  export GSPLAT_EXAMPLES=/path/to/gsplat/examples   # gsplat v1.4.0 checkout
  python scripts/train_3dgs.py default --data-dir <scene> --data-factor 1 \
      --result-dir <scene>/results --disable-viewer --no-normalize-world-space
"""
import json
import os
import sys

GS_EXAMPLES = os.environ.get("GSPLAT_EXAMPLES", "")
if not os.path.isfile(os.path.join(GS_EXAMPLES, "simple_trainer.py")):
    sys.exit("Set GSPLAT_EXAMPLES to the examples/ folder of a gsplat v1.4.0 checkout.")
sys.path.insert(0, GS_EXAMPLES)

import numpy as np  # noqa: E402
import tyro  # noqa: E402

import simple_trainer as st  # noqa: E402
from datasets.colmap import Dataset as _StockDataset  # noqa: E402


class SplitDataset(_StockDataset):
    def __init__(self, parser, split="train", **kwargs):
        super().__init__(parser, split=split, **kwargs)
        with open(os.path.join(parser.data_dir, "split.json")) as f:
            sp = json.load(f)
        train = set(sp.get("train_frames") or sp.get("training_frames"))
        closeups = set(sp.get("closeup_frames", []))
        names = [os.path.basename(n) for n in parser.image_names]
        idx = [i for i, n in enumerate(names) if n in train]
        assert len(idx) == len(train), f"train frames missing from COLMAP: {sorted(train - set(names))}"
        assert not ({names[i] for i in idx} & closeups), "close-up frame in train set"
        self.indices = np.array(idx)
        print(f"[train_3dgs] split={split}: {len(idx)} frames", flush=True)


st.Dataset = SplitDataset  # Runner looks up `Dataset` in simple_trainer's module globals

if __name__ == "__main__":
    # Same entry point as simple_trainer.py's __main__ block.
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            st.Config(strategy=st.DefaultStrategy(verbose=True), normalize_world_space=False),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            st.Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=st.MCMCStrategy(verbose=True),
                normalize_world_space=False,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    st.cli(st.main, cfg, verbose=True)
