"""Raw Opus p2 probabilities of all 811 reads with their wave (Master 10-20 16:4x: wave 2 = the 252 reads saved after 16:01 = the cleanest
replication set, no training target used them; OAI: no; CPU). data/v2/opus_raw.csv: StudyInstanceUID, Fracture, Contusion, PF OA (raw Opus,
build_p2_v2.reads = the same reader as every target), wave (1 = the frozen 559, asserted == data/image_labels/opus_p2.json selection.wave).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_opus_raw_v2.py"""
import json
from pathlib import Path

from build_p2_v2 import REPO, reads

EXP = Path(__file__).resolve().parents[1]


def main():
    T = reads()
    wave = {k: v["selection"]["wave"] for k, v in json.load(open(REPO / "data/image_labels/opus_p2.json")).items()}
    assert set(wave) == set(T.index), "opus_p2.json != the read files"
    T["wave"] = T.index.map(wave)
    fz = __import__("pandas").read_csv(EXP / "data/ext_p2_main.csv").StudyInstanceUID
    assert set(T.index[T.wave == 1]) == set(fz), "wave 1 != the frozen 559"
    T.to_csv(EXP / "data/v2/opus_raw.csv")
    print(f"OAI: no. data/v2/opus_raw.csv: {len(T)} reads, wave counts {T.wave.value_counts().to_dict()}; wave-2 p >= .5 "
          + ", ".join(f"{c} {int((T.loc[T.wave == 2, c] >= .5).sum())}" for c in ("Fracture", "Contusion", "PF OA")))


if __name__ == "__main__":
    main()
