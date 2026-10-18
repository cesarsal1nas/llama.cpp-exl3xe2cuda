#!/usr/bin/env python3
# BF16 GGUF from NVIDIA's Nemotron 3.5 Bolt BF16 checkpoint (text model only).
# The checkpoint wraps the language model in an omni config, so hparams come from llm_config,
# tensor names lose the language_model. prefix, and the tokenizer and nemotron_h_moe metadata
# are copied from NVIDIA's 24GB GGUF so both files describe the same model.
#
#   scripts/bolt/convert-bf16.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/ \
#       --metadata-from NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
#       --outtype bf16 --outfile Bolt-BF16.gguf
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gguf-py")]

if "--metadata-from" not in sys.argv:
    sys.exit("--metadata-from <NVIDIA 24GB GGUF> is required")
i = sys.argv.index("--metadata-from")
METADATA = Path(sys.argv[i + 1])
del sys.argv[i:i + 2]

import conversion  # noqa: E402
from conversion.base import ModelBase  # noqa: E402
from conversion.nemotron import NemotronHPuzzleModel  # noqa: E402


@ModelBase.register("NemotronH_Omni_Reasoning_V3", "NemotronHPuzzleForCausalLM")
class BoltOmniText(NemotronHPuzzleModel):
    model_arch = NemotronHPuzzleModel.model_arch

    def __init__(self, dir_model, *args, **kwargs):
        outer = json.loads((Path(dir_model) / "config.json").read_text())
        inner = dict(outer["llm_config"])
        inner.setdefault("n_routed_experts", inner.get("num_experts", inner.get("num_local_experts")))
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if key in outer:
                inner[key] = outer[key]
        kwargs["hparams"] = inner
        super().__init__(dir_model, *args, **kwargs)
        self._ssm_layers = [i for i, b in enumerate(self.block_configs) if b["block_type"] == "mamba"]
        self._mlp_layers = [i for i, b in enumerate(self.block_configs) if b["block_type"] == "moe"]

    def copy_metadata(self, prefix):
        import gguf
        reader = gguf.GGUFReader(METADATA)
        for key, field in reader.fields.items():
            if key.startswith(prefix):
                self.gguf_writer.add_key_value(key, field.contents(), field.types[0],
                                               field.types[1] if len(field.types) > 1 else None)

    def set_gguf_parameters(self):
        self.copy_metadata("nemotron_h_moe.")
        self.gguf_writer.add_file_type(int(self.ftype))

    def set_vocab(self):
        self.copy_metadata("tokenizer.")

    def modify_tensors(self, data_torch, name, bid):
        import torch
        if name.endswith("mixer.A_log"):
            yield from ModelBase.modify_tensors(self, -torch.exp(data_torch.float()).reshape(-1, 1), name, bid)
            return
        old = self.hparams["n_routed_experts"]
        if bid is not None and self.block_configs[bid]["block_type"] == "moe":
            self.hparams["n_routed_experts"] = self.block_configs[bid]["n_routed_experts"]
        try:
            yield from super().modify_tensors(data_torch, name, bid)
        finally:
            self.hparams["n_routed_experts"] = old

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith("language_model."):
            return None
        return super().filter_tensors((name.removeprefix("language_model."), gen))


conversion.TEXT_MODEL_MAP["NemotronH_Omni_Reasoning_V3"] = "nemotron"
runpy.run_path(str(ROOT / "convert_hf_to_gguf.py"), run_name="__main__")
