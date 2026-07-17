# WaterFlow Style Guide

Derived from a pass over `main` (aa8b771). Descriptive, not aspirational: this records what
the codebase *does* so new code matches it. Local scratch file — gitignored, not committed.

## Backwards compatibility: don't

**Do not write backwards-compatibility shims.** No fallbacks for old cache layouts, no
tolerating stale on-disk artifacts, no deprecation paths for renamed params. Stated
directly by the author.

Prefer a **loud failure over a silent degrade**. If a change invalidates caches or
checkpoints, let the old artifact raise and make the user regenerate it — that is the
desired behavior, not a defect to paper over. A `cached.get(key, <default>)` that lets a
stale artifact load as though it were current is exactly the wrong move: it converts a
crash into silently wrong data.

Corollaries:
- Removing a default (e.g. a hardcoded path) is fine; every caller passes it anyway.
- Widening a return type / changing a signature is fine — fix the callers, don't overload.
- Note that `../local_flow` *does* carry some compat shims. Don't import them on parity
  grounds; parity is about behavior, not about inheriting its legacy handling.
- Do still **flag** cache/checkpoint invalidation in the PR description for reviewer
  sign-off — the roadmap requires it. Flagging ≠ shimming.

## Tooling

- Python `>=3.12,<3.13`. Use 3.12 features freely (PEP 695 `type` aliases, `itertools.batched`).
- **ruff** (`check --fix` + `format`) and **ty** run via pre-commit. `mypy` block in
  `pyproject.toml` and `requirements-dev.txt` (black/flake8/isort/mypy/hypothesis) are stale — ignore them.
- `select = ["E", "F", "I001", "UP"]`; `E501` ignored so there's no hard line-length error, but
  ruff-format still wraps at **88**. Write code at 88; long docstring lines are tolerated.
- `UP` is what enforces `X | None` and builtin generics — don't fight it.

## Typing

- **Builtin generics only**: `list[str]`, `dict[str, float]`, `tuple[int, int]`. Zero uses of
  `typing.List`/`Dict`/`Tuple` in the repo.
- **`X | None`, never `Optional[X]`.**
- `from __future__ import annotations` in `src/` modules, placed **after** the module docstring.
- Tensors: `torch.Tensor` in most signatures; bare `Tensor` (`from torch import Tensor`) is used
  in a few short ones. NumPy is `np.ndarray` — no `NDArray`.
- Shapes go in **docstrings**, not types (`jaxtyping` is used in exactly one script).
- `Literal` for string enums: `pool_aggr: Literal["mean", "sum", "max"] = "mean"`.
- **No dataclasses, TypedDict, Protocol, or NamedTuple anywhere.** Abstraction is `ABC` +
  `@abstractmethod`; config is a plain `dict`.
- Annotate params and returns on new code (76%/62% coverage overall, and the gaps are localized:
  `src/gvp.py` is vendored and untyped, `scripts/train.py`'s argparse helpers).

## Docstrings

- **Google style** (`Args:` / `Returns:` / `Raises:` / `Notes:`), 94% coverage — including private
  helpers and test methods. `"""` on its own line, summary on the next.
- Shapes documented inline with the arg: `src_pos: (N_src, 3) source node positions`.
- `Returns:` often names values and aligns with `—`:
  ```
  Returns:
      s: (N, embedding_dim) — raw embeddings
      V: (N, 0, 3)         — empty vector features
  ```
- `Notes:` for non-obvious library choices. `__init__` docstrings often start straight at `Args:`.
- Module docstrings enumerate contents with a `This module provides:` list.
- One-liners for trivial functions. No reST — that's only in vendored `src/gvp.py`.

## Comments

- **Low density in `src/`** (~2-7%), lowercase, terse, on the line above the block:
  `# remove self-edges if homogeneous`.
- **Trailing shape comments** are a signature pattern in tensor code:
  ```python
  pos = data["protein"].pos          # (N_total, 3)
  mean_pos = scatter_mean(pos, batch_p, dim=0)   # (num_graphs, 3)
  ```
- `NOTE:` for hard-won why. `TODO:` as multi-line prose at the top of a function body.
- **Section dividers only in tests**, never in `src/`, exact form (two blank lines around):
  `# ============== Shared encoder fixtures ==============`
- Don't add a `# module_name.py` header comment — 14/20 files omit it.

## Naming

- `snake_case` / `CamelCase`; acronyms stay upper in class names (`GVPEncoder`).
- `_prefix` for module-private helpers and lazily-inferred state (`self._embedding_dim` behind a
  `@property`). Private names are still imported directly by tests.
- **Constants are `UPPER_SNAKE` in `src/constants.py`** (`NUM_RBF`, `RBF_CUTOFF`, `EDGE_PP`…) and
  used as default argument values: `radius: float = RBF_CUTOFF`. Script/test-scoped constants may
  live module-level in that file.
- Domain abbreviations to reuse: `res_id`, `ins_code`, `coords`/`pos`, `idx`, `asu`, `bts`,
  `pp`/`ww`/`pw`/`wp`, `com`, `edia`, `rbf`, `knn`, `s`/`V`, `src`/`dst`, `emb`, `ckpt`,
  `pdb_id`, `cache_key`, `mates`, `het`.
- Math-y capitals (`D`, `X1`, `V`) are fine in numeric code — `E741` is deliberately ignored.

## Imports

- Three groups (stdlib / third-party / first-party), **two blank lines after the block**
  (`lines-after-imports = 2`).
- `order-by-type = false` ⇒ names sort **case-insensitively, classes interleaved with functions**:
  `from src.gvp import EdgeUpdate, GVP, GVPConvLayer`.
- **Absolute `src.` imports only** — zero relative imports in the repo. Tests rely on
  `pythonpath = .`.
- `from x import y` dominates; bare `import x` is for stdlib and aliased third-party.
- Fixed aliases: `np`, `nn`, `F`, `plt`, `pd`, `bts` (biotite.structure), `spdist`
  (scipy.spatial.distance). `from tqdm import tqdm` (not `tqdm.auto`) dominates 4:1.
- **Function-local imports are deliberate**, not sloppy: breaking cycles, keeping heavy/optional
  deps out of module scope (`loguru` in `src/utils.py`, `sklearn` in `compute_placement_metrics`
  — sklearn is undeclared in deps, so that local import is load-bearing). In tests/conftest,
  local imports inside fixture bodies are the norm.

## Structure

- Median function ~31 lines; long procedural functions tolerated in `scripts/` and preprocessing.
- Class layout: docstring → `__init__` → `@property` → `@staticmethod` → public → private `_`.
- **No Lightning, no hydra/omegaconf.** Training is a hand-rolled loop in `scripts/train.py`.
- Config passing: long explicit keyword params with defaults (`ProteinWaterDataset.__init__` has
  25), plus a `dict`-based registry/factory for encoders (`build_encoder` → `cls.from_config`).
  `from_config` reads required keys as `config["k"]`, optional as `config.get("k", default)`.
- Encoders register via stacked `@register_encoder("name")` decorators; registration fires as an
  **import side effect** from `src/__init__.py`. A new encoder must be imported there.
- Abstract methods `raise NotImplementedError("Subclasses must implement forward")`, not `...`.

## Errors and logging

- **`raise`, never `assert`** — zero asserts in `src/`/`scripts/` (tests use them freely).
- `ValueError` (×24) for bad values, `KeyError` (×6) for missing registry/dict keys,
  `FileNotFoundError` (×5), `RuntimeError` for illegal lifecycle.
- Message style: f-string, offending value in single quotes, then the expectation — often two
  implicitly-concatenated literals:
  ```python
  raise ValueError(
      f"Unsupported encoder_type '{self.encoder_type}'. "
      "Expected one of: gvp, slae, esm"
  )
  ```
- `try/except` is rare (11 blocks) and only at I/O / third-party boundaries. Shapes:
  log-and-degrade (return sentinel), log-and-collect-failures (append to `failures` list, write a
  failure log), or narrow tuple catch. `except Exception as e` dominates.
- **loguru only** — never stdlib `logging`, never `print`. `logger.exception` (not `.error`) inside
  `except` when the traceback matters. Don't prefix `"Warning: "` inside `logger.warning` (two
  legacy sites do; it's a wart).
- `setup_logging_for_tqdm` in `src/utils.py` is the centralized setup; new scripts call it.

## Tests

- **Classes dominate**: 64 `Test*` classes / 268 methods vs 17 bare functions. `class TestSubject:`
  with a one-line docstring.
- `test_<behavior>` names + a one-line docstring phrased as an expectation:
  `"""Unknown elements should go to the 'other' bucket (last index)."""`
- Marks: `unit` (×47, applied at class level), `integration`, `slow`. `--strict-markers` is on, so
  **only those three are legal — no `xfail`**.
- Fixtures in `tests/conftest.py` (10) or at module top under a divider; all have docstrings.
  Factory-fixture pattern (`create_mock_dataset` returns an inner `_create(...)`).
- Devices: the `device` fixture is CPU-safe (`cuda if available else cpu`). CUDA-only tests guard
  with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`. Seed with `torch.manual_seed(0)`
  plus `torch.cuda.manual_seed_all(0)` under an availability check.
- Test data resolves through `conftest.py`'s `PDB_BASE_DIR` / `_resolve_pdb_path` — never hardcode
  paths. Those helpers **raise `FileNotFoundError` rather than skip**. Temp files use `tmp_path`.
- Assertions: plain `assert`; `torch.allclose`/`pytest.approx` for numerics. Exception tests always
  pin the message: `pytest.raises(ValueError, match="Unsupported encoder_type")`.
- `parametrize` is under-used (5 total, all in `test_dataset.py`) — fine to add.
- Each PDB fixture's docstring says *why that PDB*: `"""8dzt - fails water clash check at 2%."""`

## Formatting

- **Double quotes universally** in code (zero single-quoted literals). Prose/docstrings refer to
  string values in single quotes: `encoder_type: 'gvp', 'slae', or 'esm'`.
- Magic trailing commas everywhere ruff-format has exploded a call/signature — keep them.
- 2 blank lines between top-level defs, 1 between methods, 2 after imports.

## Traps

1. `src/utils.py`'s module docstring is **dead** — `from __future__ import annotations` sits above
   it. Don't replicate; docstring first.
2. **`build_knn_edges` is duplicated verbatim** in `src/utils.py:33` and `src/flow.py:29`.
   `flow.py` does not import the utils copy. Touch one, check the other.
3. `sklearn` is used but undeclared in deps — its function-local import is why nothing breaks.
4. `src/gvp.py` is vendored (reST docstrings, untyped). Don't fix its style; don't copy it.
5. `tests/test_train_config.py` is a consistency guard: adding a param to
   `ProteinWaterDataset.__init__` likely needs a matching `train.py` argparse default.
6. Hardcoded lab paths exist in signatures (`/sb/wankowicz_lab/...`) with TODOs to remove — don't add more.
7. A bare `pytest` writes `htmlcov/` (coverage is on by default via `addopts`).
