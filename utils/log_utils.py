import csv
import os

import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from utils.general_utils import resolve_detail_count_choices


def tensorboard_log_image(log_writer, tag: str, image_tensor, step):
    log_writer.experiment.add_image(
        tag,
        image_tensor,
        step,
    )


def wandb_log_image(log_writer, tag: str, image_tensor, step):
    image_dict = {
        tag: wandb.Image(image_tensor),
    }
    log_writer.experiment.log(
        image_dict,
        step=step,
    )



class PerformanceMetricLogger:
    TIMING_FIELDS = [
        "training_start_time",
        "metric_start_time",
        "metric_end_time",
        "metric_elapsed_sec",
        "elapsed_since_training_start_sec",
    ]
    METRIC_FIELDS = [
        "l1_loss",
        "psnr",
        "ssim",
        "lpips",
    ]
    FIELDNAMES = [
        "iteration",
        "split",
        "camera_count",
    ] + TIMING_FIELDS + METRIC_FIELDS

    def __init__(self, log_dir, interval=10000):
        self.interval = int(interval or 0)
        self.log_dir = os.path.join(log_dir, "metric_logs")
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        self._ensure_csv_path()

    def should_log(self, iteration):
        return self.interval > 0 and int(iteration) % self.interval == 0

    def log(self, iteration, split, camera_count, metrics, timing=None):
        row = {
            "iteration": int(iteration),
            "split": split,
            "camera_count": int(camera_count),
        }
        timing = timing or {}
        for field in self.TIMING_FIELDS:
            row[field] = self._to_scalar(timing.get(field, ""))
        for field in self.METRIC_FIELDS:
            row[field] = self._to_scalar(metrics[field])

        with open(self.csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

    def plot(self):
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if not rows:
            return

        self._plot_panels(
            rows,
            self.METRIC_FIELDS,
            "training_metrics_quality.png",
            "Metric value",
        )
        self._plot_panels(
            rows,
            [
                "metric_elapsed_sec",
                "elapsed_since_training_start_sec",
            ],
            "training_metrics_timing.png",
            "Seconds",
        )

    def _ensure_csv_path(self):
        if not os.path.exists(self.csv_path):
            self._write_header(self.csv_path)
            return
        if self._csv_header_matches(self.csv_path):
            return

        stem = "training_metrics"
        candidate = os.path.join(self.log_dir, stem + "_1.csv")
        index = 2
        while os.path.exists(candidate) and not self._csv_header_matches(candidate):
            candidate = os.path.join(self.log_dir, "{}_{}.csv".format(stem, index))
            index += 1
        self.csv_path = candidate
        if not os.path.exists(self.csv_path):
            self._write_header(self.csv_path)

    def _csv_header_matches(self, csv_path):
        with open(csv_path, "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                return next(reader) == self.FIELDNAMES
            except StopIteration:
                return False

    def _write_header(self, csv_path):
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def _plot_panels(self, rows, fields, filename, ylabel):
        split_rows = {}
        for row in rows:
            split_rows.setdefault(row.get("split", "unknown"), []).append(row)
        for split in split_rows:
            split_rows[split].sort(key=lambda row: int(row.get("iteration", 0) or 0))

        column_count = 2
        row_count = (len(fields) + column_count - 1) // column_count
        fig = Figure(figsize=(12, 4 * row_count), dpi=150)
        canvas = FigureCanvasAgg(fig)
        for index, field in enumerate(fields):
            ax = fig.add_subplot(row_count, column_count, index + 1)
            for split, split_data in split_rows.items():
                if field not in split_data[0]:
                    continue
                iterations = [int(row.get("iteration", 0) or 0) for row in split_data]
                values = [float(row.get(field, 0) or 0) for row in split_data]
                ax.plot(iterations, values, marker="o", linewidth=1.5, markersize=3, label=split)
            ax.set_title(field)
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        for index in range(len(fields), row_count * column_count):
            ax = fig.add_subplot(row_count, column_count, index + 1)
            ax.axis("off")

        fig.tight_layout()
        canvas.print_png(os.path.join(self.log_dir, filename))

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                return value.item()
            return value.cpu().tolist()
        return value


class TrainingResourceLogger:
    BASE_FIELDS = [
        "iteration",
        "render_gaussian_budget",
        "visible_anchor_count",
        "detail_budget",
        "selected_detail_count",
        "budget_overflow",
        "base_render_gaussian_budget",
        "effective_render_gaussian_budget",
        "stage_budget_scale",
        "stage_budget_reference_vram_mb",
        "stage_budget_observed_vram_mb",
    ]
    VRAM_FIELDS = [
        "vram_allocated_mb",
        "vram_reserved_mb",
        "vram_max_allocated_mb",
        "vram_max_reserved_mb",
        "vram_free_mb",
        "vram_total_mb",
    ]

    def __init__(self, log_dir, detail_max_slots, detail_count_choices=None):
        self.log_dir = os.path.join(log_dir, "resource_logs")
        self.csv_path = os.path.join(self.log_dir, "training_resources.csv")
        self.detail_max_slots, _, self.detail_count_choices = resolve_detail_count_choices(
            detail_max_slots, detail_count_choices
        )
        self.detail_fields = [
            "detail_count_{}".format(slot_count)
            for slot_count in self.detail_count_choices
        ]
        self.fieldnames = self.BASE_FIELDS + self.detail_fields + self.VRAM_FIELDS
        os.makedirs(self.log_dir, exist_ok=True)
        self._ensure_compatible_csv_path()

    def _ensure_compatible_csv_path(self):
        if self._csv_header_matches(self.csv_path):
            return
        if not os.path.exists(self.csv_path):
            self._write_header(self.csv_path)
            return

        stem = "training_resources_choices_{}".format(
            "_".join(str(choice) for choice in self.detail_count_choices)
        )
        candidate = os.path.join(self.log_dir, stem + ".csv")
        index = 1
        while os.path.exists(candidate) and not self._csv_header_matches(candidate):
            candidate = os.path.join(self.log_dir, "{}_{}.csv".format(stem, index))
            index += 1
        self.csv_path = candidate
        if not os.path.exists(self.csv_path):
            self._write_header(self.csv_path)

    def _csv_header_matches(self, csv_path):
        if not os.path.exists(csv_path):
            return False
        with open(csv_path, "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                return next(reader) == self.fieldnames
            except StopIteration:
                return False

    def _write_header(self, csv_path):
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, iteration, budget_stats, detail_counts):
        row = {field: 0 for field in self.fieldnames}
        row["iteration"] = int(iteration)
        for field in self.BASE_FIELDS[1:]:
            row[field] = self._to_scalar(budget_stats.get(field, 0))
        row["budget_overflow"] = int(bool(row["budget_overflow"]))

        for field, value in zip(self.detail_fields, self._detail_distribution(detail_counts)):
            row[field] = int(value)

        row.update(self._vram_stats())
        with open(self.csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(row)

    def plot(self):
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if not rows:
            return

        self._plot_lines(
            rows,
            [
                "render_gaussian_budget",
                "visible_anchor_count",
                "detail_budget",
                "selected_detail_count",
                "base_render_gaussian_budget",
                "effective_render_gaussian_budget",
            ],
            "training_resources_budget.png",
            "Gaussian count",
        )
        self._plot_lines(
            rows,
            self.detail_fields,
            "training_resources_detail_distribution.png",
            "Anchor count",
        )
        self._plot_lines(
            rows,
            self.VRAM_FIELDS + [
                "stage_budget_reference_vram_mb",
                "stage_budget_observed_vram_mb",
            ],
            "training_resources_vram.png",
            "MiB",
        )

    def _detail_distribution(self, detail_counts):
        if detail_counts is None:
            return [0] * len(self.detail_count_choices)
        detail_counts = detail_counts.detach().to(dtype=torch.long)
        return [int((detail_counts == slot_count).sum().item()) for slot_count in self.detail_count_choices]

    def _vram_stats(self):
        stats = {field: 0.0 for field in self.VRAM_FIELDS}
        if not torch.cuda.is_available():
            return stats

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        try:
            free, total = torch.cuda.mem_get_info()
        except RuntimeError:
            free, total = 0, 0

        denom = 1024.0 ** 2
        stats.update({
            "vram_allocated_mb": allocated / denom,
            "vram_reserved_mb": reserved / denom,
            "vram_max_allocated_mb": max_allocated / denom,
            "vram_max_reserved_mb": max_reserved / denom,
            "vram_free_mb": free / denom,
            "vram_total_mb": total / denom,
        })
        return stats

    def _plot_lines(self, rows, fields, filename, ylabel):
        iterations = [int(row["iteration"]) for row in rows]
        fig = Figure(figsize=(12, 6), dpi=150)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for field in fields:
            if field not in rows[0]:
                continue
            values = [float(row.get(field, 0) or 0) for row in rows]
            ax.plot(iterations, values, label=field)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        canvas.print_png(os.path.join(self.log_dir, filename))

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                return value.item()
            return value.cpu().tolist()
        return value
