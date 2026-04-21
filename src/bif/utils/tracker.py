"""SwanLab experiment tracking integration.

Two integration paths:
  1. Manual: init_run() + log() + finish() — used by BIF runner, analysis, pipeline
  2. HF Trainer: create_hf_callback() — used by checkpoint_trainer, schedule_trainer

Pipeline mode: when SWANLAB_PIPELINE_RUN_ID env var is set, all steps share
ONE SwanLab run via ``id=<run_id>`` + ``resume='allow'``.  The pipeline
runner calls finish_pipeline() at the very end to close the run.
"""

from __future__ import annotations

import os
import time
from typing import Any

import swanlab

_pending_logs: list[tuple[dict[str, Any], int | None]] = []
_INIT_TIMEOUT = 30

_ENV_RUN_ID = "SWANLAB_PIPELINE_RUN_ID"
_ENV_EXPERIMENT = "SWANLAB_PIPELINE_EXPERIMENT"
_ENV_PROJECT = "SWANLAB_PROJECT"

_DEFAULT_PROJECT = "bif"


def get_project() -> str:
    return os.environ.get(_ENV_PROJECT, _DEFAULT_PROJECT)


def _is_initialised() -> bool:
    try:
        return swanlab.get_run() is not None
    except Exception:
        return False


def _flush_pending() -> None:
    global _pending_logs
    if not _pending_logs:
        return
    remaining: list[tuple[dict[str, Any], int | None]] = []
    for data, step in _pending_logs:
        try:
            swanlab.log(data, step=step)
        except Exception:
            remaining.append((data, step))
    _pending_logs = remaining


def init_run(
    experiment_name: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    description: str = "",
    run_name: str | None = None,
) -> None:
    """Initialize a SwanLab run (call only from rank 0)."""
    global _pending_logs
    _pending_logs = []

    pipeline_run_id = os.environ.get(_ENV_RUN_ID)
    pipeline_experiment = os.environ.get(_ENV_EXPERIMENT)

    _proj = get_project()
    if pipeline_run_id and pipeline_experiment and run_name is None:
        init_kwargs: dict[str, Any] = {
            "project": _proj,
            "experiment_name": pipeline_experiment,
            "id": pipeline_run_id,
            "resume": "allow",
            "description": description,
            "config": config,
            "tags": tags,
        }
    else:
        display_name = run_name if run_name is not None else experiment_name
        init_kwargs = {
            "project": _proj,
            "experiment_name": display_name,
            "group": experiment_name,
            "description": description,
            "config": config,
            "tags": tags,
        }

    swanlab.init(**init_kwargs)
    deadline = time.monotonic() + _INIT_TIMEOUT
    while time.monotonic() < deadline:
        if _is_initialised():
            break
        time.sleep(0.5)


def create_hf_callback(
    experiment_name: str | None = None,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    description: str = "",
) -> Any:
    """Create a SwanLabCallback for HF Trainer.

    In pipeline mode, the pipeline runner has already called init_run()
    BEFORE spawning the training subprocess, so SwanLab is already
    initialised.  We pass mode="disabled" to prevent SwanLabCallback
    from calling swanlab.init() again — all logging goes through the
    existing run via our manual log() calls in _GradNormCallback and
    ReplayTrainer.log().

    In standalone mode, we let SwanLabCallback manage the full lifecycle
    (init on train_begin, finish on train_end).
    """
    from swanlab.integration.transformers import SwanLabCallback

    pipeline_run_id = os.environ.get(_ENV_RUN_ID)
    pipeline_experiment = os.environ.get(_ENV_EXPERIMENT)

    if pipeline_run_id and pipeline_experiment and _is_initialised():
        return SwanLabCallback(mode="disabled")
    else:
        display_name = run_name or experiment_name or "train"
        init_kwargs: dict[str, Any] = {
            "project": get_project(),
            "experiment_name": display_name,
            "description": description,
        }
        if config:
            init_kwargs["config"] = config
        if tags:
            init_kwargs["tags"] = tags
        return SwanLabCallback(**init_kwargs)


def log(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics. Buffers calls made before init completes."""
    if not _is_initialised():
        _pending_logs.append((data, step))
        return
    _flush_pending()
    try:
        swanlab.log(data, step=step)
    except Exception:
        pass


def finish() -> None:
    """Close the SwanLab run.

    In pipeline mode, skip finish() so the next step can resume.
    Call finish_pipeline() instead at the very end of the pipeline.
    """
    if not _is_initialised():
        return
    _flush_pending()
    if os.environ.get(_ENV_RUN_ID):
        return
    try:
        swanlab.finish()
    except Exception:
        pass


def finish_pipeline() -> None:
    """Force-close the SwanLab run at the end of a pipeline.

    Unlike finish(), this ALWAYS closes the run even in pipeline mode.
    """
    if not _is_initialised():
        return
    _flush_pending()
    try:
        swanlab.finish()
    except Exception:
        pass


def log_image(key: str, path: str) -> None:
    """Log an image from a file path.  Prefer log_figure() for matplotlib figures."""
    if not _is_initialised():
        return
    try:
        swanlab.log({key: swanlab.Image(path)})
    except Exception:
        pass


def log_figure(key: str, fig: "matplotlib.figure.Figure") -> None:  # noqa: F821
    """Log a matplotlib Figure directly to SwanLab without saving to disk.

    SwanLab accepts ``swanlab.Image`` constructed from a PIL image, so we
    render the figure into an in-memory PNG buffer and hand it off — no
    temporary file is created.

    Args:
        key:  The metric name shown in SwanLab (e.g. ``"loss_curve"``).
        fig:  A matplotlib Figure object.  The figure is closed after logging
              so callers do not need to call ``plt.close()`` themselves.
    """
    if not _is_initialised():
        return
    try:
        import io
        import matplotlib  # noqa: F401 — ensure it is importable
        from PIL import Image as PILImage

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()  # copy to detach from buffer
        buf.close()
        swanlab.log({key: swanlab.Image(pil_img)})
    except Exception:
        pass
    finally:
        try:
            import matplotlib.pyplot as _plt

            _plt.close(fig)
        except Exception:
            pass


def log_heatmap(
    key: str,
    xaxis: list[str],
    yaxis: list[str],
    matrix: "np.ndarray",  # noqa: F821  shape (len(yaxis), len(xaxis))
    value_label: str = "value",
    precision: int = 4,
) -> None:
    """Log a 2-D matrix as a native SwanLab echarts HeatMap.

    Args:
        key:         Metric name shown in SwanLab.
        xaxis:       Labels for the x-axis (columns).
        yaxis:       Labels for the y-axis (rows).
        matrix:      2-D array of shape (len(yaxis), len(xaxis)).
        value_label: Series name displayed in the tooltip.
        precision:   Decimal places to round values to.
    """
    if not _is_initialised():
        return
    try:
        # echarts HeatMap expects value as list of [x_idx, y_idx, val]
        value = [
            [j, i, round(float(matrix[i, j]), precision)]
            for i in range(len(yaxis))
            for j in range(len(xaxis))
        ]
        chart = swanlab.echarts.HeatMap()
        chart.add_xaxis(xaxis)
        chart.add_yaxis(value_label, yaxis, value)
        swanlab.log({key: chart})
    except Exception:
        pass


def log_bar(
    key: str,
    xaxis: list[str],
    series: dict[str, list],
    stack: bool = False,
) -> None:
    """Log a bar chart as a native SwanLab echarts Bar.

    Args:
        key:    Metric name shown in SwanLab.
        xaxis:  Category labels for the x-axis.
        series: Dict mapping series name → list of y values (same length as xaxis).
        stack:  If True, all series are stacked onto each other.
    """
    if not _is_initialised():
        return
    try:
        chart = swanlab.echarts.Bar()
        chart.add_xaxis(xaxis)
        for name, values in series.items():
            chart.add_yaxis(
                name,
                [round(float(v), 4) if v is not None else None for v in values],
                stack="stack0" if stack else None,
            )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_line(
    key: str,
    xaxis: list[str],
    series: dict[str, list],
    smooth: bool = False,
) -> None:
    """Log a multi-series line chart as a native SwanLab echarts Line.

    Args:
        key:    Metric name shown in SwanLab.
        xaxis:  X-axis labels (e.g. checkpoint names).
        series: Dict mapping series name → list of y values.
        smooth: If True, render as smooth curves by default (user can toggle).
    """
    if not _is_initialised():
        return
    try:
        from pyecharts.options.global_options import (
            ToolboxOpts,
            ToolBoxFeatureOpts,
            ToolBoxFeatureDataZoomOpts,
            ToolBoxFeatureMagicTypeOpts,
            ToolBoxFeatureRestoreOpts,
        )

        chart = swanlab.echarts.Line()
        chart.add_xaxis(xaxis)
        for name, values in series.items():
            chart.add_yaxis(
                name,
                [round(float(v), 6) if v is not None else None for v in values],
                is_smooth=smooth,
                is_symbol_show=False,
            )
        chart.set_global_opts(
            toolbox_opts=ToolboxOpts(
                feature=ToolBoxFeatureOpts(
                    magic_type=ToolBoxFeatureMagicTypeOpts(
                        type_=["line", "bar", "stack"],
                    ),
                    data_zoom=ToolBoxFeatureDataZoomOpts(
                        is_show=True,
                    ),
                    restore=ToolBoxFeatureRestoreOpts(
                        is_show=True,
                    ),
                ),
            ),
        )
        swanlab.log({key: chart})
    except Exception:
        pass


def log_table(
    key: str,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    """Log a table as a native SwanLab echarts Table.

    Useful for displaying top-K samples with text previews, scores,
    and metadata.

    Args:
        key:     Metric name shown in SwanLab (e.g. ``"top_samples"``).
        headers: Column names.
        rows:    List of rows, each row is a list of values matching headers.
    """
    if not _is_initialised():
        return
    try:
        table = swanlab.echarts.Table()
        str_rows = []
        for row in rows:
            str_rows.append([str(v) if v is not None else "" for v in row])
        table.add(headers=headers, rows=str_rows)
        swanlab.log({key: table})
    except Exception:
        pass
