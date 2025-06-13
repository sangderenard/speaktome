class EmbeddingLoader:
    """Load embeddings from various formats for PCA processing."""

    def __init__(self, source: str, ids_file: str | None = None) -> None:
        self.source = source
        self.ids_file = ids_file

    def load(self):
        """Return embeddings as an ``(N, D)`` array."""
        import numpy as np
        from pathlib import Path

        path = Path(self.source)
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix in {".pt", ".pth"}:
            import torch

            return torch.load(path, map_location="cpu").numpy()
        if path.suffix == ".json":
            import json

            data = json.loads(path.read_text())
            return np.asarray(data, dtype=float)
        if path.suffix == ".csv":
            return np.loadtxt(path, delimiter=",")
        raise ValueError(f"Unsupported embedding format: {path.suffix}")

    def item_ids(self):
        from pathlib import Path

        if self.ids_file and Path(self.ids_file).is_file():
            return [line.strip() for line in Path(self.ids_file).read_text().splitlines()]
        return [str(i) for i in range(self.load().shape[0])]


def compute(X, *, scale: bool = False):
    """Return coordinates along PCA1 and explained variance."""
    import numpy as np

    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    if scale:
        Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-9)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    axis = vt[0]
    coords = Xc.dot(axis)
    total_var = (Xc ** 2).sum()
    variance = float((s[0] ** 2) / total_var) if total_var else 0.0
    return coords, variance


def plot_spectrum(coords, ids, *, outfile: str | None = None):
    """Return a matplotlib figure showing items along the spectrum."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from exc

    coords = list(coords)
    ids = list(ids)

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.scatter(coords, [0] * len(coords), marker="|")
    for c, label in zip(coords, ids):
        ax.text(c, 0, label, rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Semantic Spectrum")
    if outfile:
        fig.savefig(outfile, bbox_inches="tight")
    return fig


# ########## STUB: semantic_spectrum_rest ##########
# PURPOSE: Provide a FastAPI wrapper exposing ``compute`` and ``plot_spectrum``.
# EXPECTED BEHAVIOR: Serve ``POST /compute`` and ``GET /spectrum.png`` endpoints.
# INPUTS: Embedding files via multipart or URL.
# OUTPUTS: JSON coordinates and optional PNG image.
# KEY ASSUMPTIONS/DEPENDENCIES: relies on ``compute`` and ``plot_spectrum`` above.
# TODO:
#   - Add Docker configuration and integration tests.
# NOTES: This stub captures remaining work from the conceptual flag proposal.
# ###########################################################################

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import Response
    import tempfile
    import io

    app = FastAPI()
    _LATEST: dict[str, list[str] | list[float] | float] | None = None

    @app.post("/compute")
    async def compute_endpoint(
        embeddings: UploadFile = File(...),
        ids: UploadFile | None = None,
    ) -> dict[str, object]:
        """Return coordinates and variance for uploaded embeddings."""
        emb_bytes = await embeddings.read()
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(emb_bytes)
            emb_path = tf.name
        ids_path = None
        if ids is not None:
            id_bytes = await ids.read()
            with tempfile.NamedTemporaryFile(delete=False) as tf2:
                tf2.write(id_bytes)
                ids_path = tf2.name
        loader = EmbeddingLoader(emb_path, ids_file=ids_path)
        X = loader.load()
        item_ids = loader.item_ids()
        coords, var = compute(X)
        global _LATEST
        _LATEST = {"ids": item_ids, "coords": coords.tolist(), "variance": var}
        return _LATEST

    @app.get("/spectrum.png")
    async def spectrum_png() -> Response:
        if _LATEST is None:
            raise HTTPException(status_code=404, detail="No spectrum computed")
        fig = plot_spectrum(_LATEST["coords"], _LATEST["ids"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return Response(buf.getvalue(), media_type="image/png")
except Exception:  # pragma: no cover - FastAPI missing
    app = None


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute a 1D semantic spectrum via PCA")
    parser.add_argument("embeddings", help="Path to embeddings (.npy, .pt, .csv, .json)")
    parser.add_argument("--ids", help="Optional text file of item IDs", default=None)
    parser.add_argument("--outfig", help="Save spectrum plot to file", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    loader = EmbeddingLoader(args.embeddings, ids_file=args.ids)
    X = loader.load()
    ids = loader.item_ids()
    coords, var = compute(X)
    print(f"Explained variance: {var:.4f}")
    for item_id, coord in zip(ids, coords):
        print(f"{coord:.6f}\t{item_id}")
    if args.outfig:
        plot_spectrum(coords, ids, outfile=args.outfig)


if __name__ == "__main__":
    main()
