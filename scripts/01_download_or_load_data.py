import yaml
from datasets import load_dataset


def _load_one(ds_cfg: dict):
    hf_id = ds_cfg["hf_id"]
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "train")
    streaming = bool(ds_cfg.get("streaming", False))

    kwargs = {"split": split, "streaming": streaming}
    if subset:
        dset = load_dataset(hf_id, subset, **kwargs)
    else:
        dset = load_dataset(hf_id, **kwargs)
    return dset


def main():
    with open("configs/data.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("=== Dataset load smoke test ===")
    for ds in cfg["datasets"]:
        print(f"\nLoading: {ds['name']}  (hf_id={ds['hf_id']} subset={ds.get('subset')} split={ds.get('split')})")
        dset = _load_one(ds)

        # Streaming datasets don't support len() reliably
        if hasattr(dset, "features"):
            print("Features:", list(dset.features.keys())[:30])
        else:
            print("Features: <unknown>")

        try:
            print("Num rows:", len(dset))
            sample = dset[0]
        except TypeError:
            # streaming
            sample = next(iter(dset.take(1)))
            print("Num rows: <streaming>")

        print("Sample keys:", list(sample.keys()))
        print("Sample preview:", {k: sample[k] for k in list(sample.keys())[:6]})


if __name__ == "__main__":
    main()
