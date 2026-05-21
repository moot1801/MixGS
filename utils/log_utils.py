import csv
import os

import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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

    def __init__(self, log_dir, detail_max_slots):
        self.log_dir = os.path.join(log_dir, "resource_logs")
        self.csv_path = os.path.join(self.log_dir, "training_resources.csv")
        self.detail_slot_exponent = max(0, int(detail_max_slots))
        self.detail_max_slots = 1 << self.detail_slot_exponent
        self.detail_count_choices = [0] + [1 << i for i in range(self.detail_slot_exponent + 1)]
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

        stem = "training_resources_slots_{}".format(self.detail_max_slots)
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
