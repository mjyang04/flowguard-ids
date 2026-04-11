from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nids.config import TrainingConfig
from nids.evaluation.latency import measure_inference_latency
from nids.evaluation.metrics import compute_nids_metrics
from nids.training.auc_loss import pairwise_auc_loss
from nids.training.callbacks import EarlyStopping
from nids.training.focal_loss import BinaryFocalLoss
from nids.training.optimizers import build_optimizer, build_scheduler
from nids.utils.io import save_json
from nids.utils.logging import get_logger


@dataclass
class EvaluationResult:
    loss: float
    metrics: dict
    predictions: np.ndarray
    labels: np.ndarray
    scores: np.ndarray | None = None


@dataclass
class TrainingSummary:
    best_metric: float
    best_epoch: int
    best_model_path: str
    history: list[dict]
    test_metrics: dict | None = None


class Trainer:
    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer")

    def _show_tqdm(self) -> bool:
        return bool(getattr(self.config, "use_tqdm", True))

    def _show_eval_tqdm(self) -> bool:
        return bool(getattr(self.config, "show_eval_tqdm", False))

    def _emit_epoch_log(self, message: str) -> None:
        # Use tqdm-aware output to avoid progress-bar line corruption in terminal.
        if self._show_tqdm():
            tqdm.write(message, file=sys.stdout)
        else:
            self.logger.info(message)

    def _tqdm_bar_format(self) -> str:
        # Keep second-level timing and show speed (e.g. 167.72s/it).
        return "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint_last.pt"

    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        history: list[dict],
        best_metric: float,
        best_epoch: int,
        best_model_path: Path,
    ) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_model_path": str(best_model_path),
            "selection_metric": self.config.selection_metric,
        }
        torch.save(payload, self._checkpoint_path())

    def _format_nids_metrics(self, metrics: dict) -> str:
        return (
            "avg_attack_recall={avg_attack_recall:.4f} "
            "attack_macro_precision={attack_macro_precision:.4f} "
            "benign_false_alarm_rate={benign_false_alarm_rate:.4f} "
            "mcc={mcc:.4f} "
            "ece={ece:.4f}"
        ).format(
            avg_attack_recall=float(metrics.get("avg_attack_recall", 0.0)),
            attack_macro_precision=float(metrics.get("attack_macro_precision", 0.0)),
            benign_false_alarm_rate=float(metrics.get("benign_false_alarm_rate", 0.0)),
            mcc=float(metrics.get("mcc", 0.0)),
            ece=float(metrics.get("ece", 0.0)),
        )

    def _log_training_setup(
        self,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.logger.info(
            "Training setup | device=%s num_classes=%s epochs=%s batch_size=%s train_batches=%s val_batches=%s",
            device,
            num_classes,
            self.config.num_epochs,
            getattr(train_loader, "batch_size", None),
            len(train_loader),
            len(val_loader),
        )
        self.logger.info(
            "Optimizer=%s lr=%.6f weight_decay=%.6f scheduler=%s(use=%s) early_stopping(use=%s) amp=%s grad_clip=%.2f selection_metric=%s",
            optimizer.__class__.__name__,
            optimizer.param_groups[0]["lr"],
            self.config.weight_decay,
            self.config.scheduler,
            getattr(self.config, "use_scheduler", True),
            getattr(self.config, "use_early_stopping", True),
            self.config.amp and device.type == "cuda",
            self.config.gradient_clip,
            self.config.selection_metric,
        )

    def _build_criterion(
        self, num_classes: int, class_weights: np.ndarray | None, device: torch.device
    ) -> torch.nn.Module:
        loss_type = getattr(self.config, "loss_type", "bce")
        label_smoothing = getattr(self.config, "label_smoothing", 0.0)

        if num_classes == 2:
            pos_weight = None
            if class_weights is not None and len(class_weights) >= 2 and class_weights[0] > 0:
                pos_weight = torch.tensor(
                    [float(class_weights[1] / class_weights[0])], dtype=torch.float32, device=device
                )

            if loss_type == "focal":
                self.logger.info(
                    "Using Focal Loss | alpha=%.2f gamma=%.2f",
                    self.config.focal_alpha, self.config.focal_gamma,
                )
                return BinaryFocalLoss(
                    alpha=self.config.focal_alpha,
                    gamma=self.config.focal_gamma,
                    pos_weight=pos_weight,
                )
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return torch.nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
        )

    def _predict_from_logits(self, logits: torch.Tensor, num_classes: int) -> torch.Tensor:
        if num_classes == 2:
            return (torch.sigmoid(logits) >= 0.5).long()
        return torch.argmax(logits, dim=1)

    def _add_auc_loss(
        self,
        loss: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        use_auc: bool,
    ) -> torch.Tensor:
        """Add pairwise AUC auxiliary loss when enabled."""
        if not use_auc:
            return loss
        auc_l = pairwise_auc_loss(
            outputs,
            labels,
            margin=self.config.auc_loss_margin,
            num_neg=self.config.auc_loss_num_neg,
        )
        return loss + self.config.auc_loss_lambda * auc_l

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int,
        scaler: torch.amp.GradScaler | None = None,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> float:
        model.train()
        total_loss = 0.0
        use_auc = self.config.use_auc_loss and num_classes == 2
        label_smoothing = getattr(self.config, "label_smoothing", 0.0)

        iterator = tqdm(
            dataloader,
            desc=f"Train {epoch}/{total_epochs}",
            leave=False,
            position=1 if self._show_tqdm() else 0,
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format=self._tqdm_bar_format(),
            unit="batch",
            unit_scale=False,
            disable=not self._show_tqdm(),
        )
        for step, (features, labels) in enumerate(iterator, start=1):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Apply label smoothing for binary targets
            if num_classes == 2 and label_smoothing > 0:
                smooth_labels = labels.float() * (1 - label_smoothing) + 0.5 * label_smoothing
            else:
                smooth_labels = labels.float() if num_classes == 2 else labels

            if scaler is not None:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = model(features)
                    loss = criterion(outputs, smooth_labels)
                    loss = self._add_auc_loss(loss, outputs, labels, use_auc)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, smooth_labels)
                loss = self._add_auc_loss(loss, outputs, labels, use_auc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()

            total_loss += float(loss.item())
            if self._show_tqdm():
                iterator.set_postfix(
                    loss=f"{(total_loss / step):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
        return total_loss / max(1, len(dataloader))

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module | None = None,
        device: torch.device | None = None,
        num_classes: int = 2,
        desc: str = "Eval",
        show_progress: bool = False,
    ) -> EvaluationResult:
        model.eval()
        if device is None:
            device = next(model.parameters()).device

        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        iterator = data_loader
        if show_progress and self._show_eval_tqdm():
            iterator = tqdm(
                data_loader,
                desc=desc,
                leave=False,
                position=1 if self._show_tqdm() else 0,
                dynamic_ncols=True,
                mininterval=1.0,
                bar_format=self._tqdm_bar_format(),
                unit="batch",
                unit_scale=False,
                disable=not self._show_tqdm(),
            )

        use_amp = self.config.amp and device.type == "cuda"
        with torch.no_grad():
            for features, labels in iterator:
                features = features.to(device)
                labels = labels.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(features)

                if criterion is not None:
                    if num_classes == 2:
                        loss = criterion(outputs, labels.float())
                    else:
                        loss = criterion(outputs, labels)
                    total_loss += float(loss.item())

                if num_classes == 2:
                    probs = torch.sigmoid(outputs).reshape(-1)
                    preds = (probs >= 0.5).long()
                    all_scores.append(probs.cpu().numpy())
                else:
                    preds = self._predict_from_logits(outputs, num_classes)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores) if all_scores else None
        if y_score is not None and np.isnan(y_score).any():
            nan_count = int(np.isnan(y_score).sum())
            self.logger.warning("NaN detected in model scores | count=%d/%d", nan_count, len(y_score))
        metrics = compute_nids_metrics(y_true, y_pred, y_score=y_score)
        loss = total_loss / max(1, len(data_loader)) if criterion is not None else 0.0
        metrics["loss"] = loss
        return EvaluationResult(
            loss=loss,
            metrics=metrics,
            predictions=y_pred,
            labels=y_true,
            scores=y_score,
        )

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 2,
        class_weights: np.ndarray | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> TrainingSummary:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = build_optimizer(
            model,
            name=self.config.optimizer,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = None
        if getattr(self.config, "use_scheduler", True):
            scheduler = build_scheduler(
                optimizer,
                name=self.config.scheduler,
                scheduler_factor=self.config.scheduler_factor,
                scheduler_patience=self.config.scheduler_patience,
                min_learning_rate=self.config.min_learning_rate,
                num_epochs=self.config.num_epochs,
            )
        criterion = self._build_criterion(num_classes, class_weights, device)
        scaler = (
            torch.amp.GradScaler("cuda")
            if self.config.amp and device.type == "cuda"
            else None
        )
        self._log_training_setup(device, train_loader, val_loader, num_classes, optimizer)

        early_stopping = None
        if getattr(self.config, "use_early_stopping", True):
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                delta=self.config.early_stopping_delta,
                mode="max",
            )
        best_epoch = 0
        best_metric = float("-inf")
        best_model_path = self.output_dir / "best_model.pt"
        history: list[dict] = []
        start_epoch = 1

        if resume_checkpoint is not None:
            checkpoint_path = Path(resume_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = (self.output_dir / checkpoint_path).resolve()
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                history = checkpoint.get("history", [])
                best_metric = float(checkpoint.get("best_metric", best_metric))
                best_epoch = int(checkpoint.get("best_epoch", best_epoch))
                loaded_best_model_path = checkpoint.get("best_model_path")
                if loaded_best_model_path:
                    best_model_path = Path(loaded_best_model_path)
                start_epoch = int(checkpoint.get("epoch", 0)) + 1
                if early_stopping is not None:
                    # Restore early-stopping state directly from checkpoint metadata
                    # instead of replaying history step-by-step, which can trigger
                    # should_stop before resumed training even begins.
                    early_stopping.best = best_metric
                    counter = 0
                    for record in reversed(history):
                        v = float(record.get("val_metric", 0.0))
                        if v > early_stopping.best + early_stopping.delta:
                            break
                        counter += 1
                    early_stopping.counter = min(counter, early_stopping.patience)
                self._emit_epoch_log(
                    f"Resume training from checkpoint={checkpoint_path} start_epoch={start_epoch}"
                )
            else:
                self._emit_epoch_log(
                    f"Resume checkpoint not found at {checkpoint_path}, training will start from scratch."
                )

        if start_epoch > self.config.num_epochs:
            self._emit_epoch_log(
                f"Checkpoint epoch already reached num_epochs ({self.config.num_epochs}); skip further training."
            )
            summary = TrainingSummary(
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_model_path=str(best_model_path),
                history=history,
            )
            save_json(asdict(summary), self.output_dir / "training_summary.json")
            return summary

        epoch_iterator = range(start_epoch, self.config.num_epochs + 1)
        if self._show_tqdm():
            epoch_iterator = tqdm(
                epoch_iterator,
                desc="Epochs",
                leave=True,
                position=0,
                dynamic_ncols=True,
                mininterval=1.0,
                bar_format=self._tqdm_bar_format(),
                unit="epoch",
                unit_scale=False,
            )

        for epoch in epoch_iterator:
            train_loss = self.train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=num_classes,
                scaler=scaler,
                epoch=epoch,
                total_epochs=self.config.num_epochs,
            )
            val_result = self.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                desc=f"Val {epoch}/{self.config.num_epochs}",
                show_progress=True,
            )

            current_metric = float(val_result.metrics.get(self.config.selection_metric, 0.0))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_result.loss,
                    "val_metric": current_metric,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_metric)
                else:
                    scheduler.step()

            if early_stopping is not None:
                improved = early_stopping.step(current_metric)
            else:
                improved = current_metric > best_metric
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            self._save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                history=history,
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_model_path=best_model_path,
            )

            self._emit_epoch_log(
                "epoch=%s train_loss=%.4f val_loss=%.4f %s=%.4f"
                % (
                    epoch,
                    train_loss,
                    val_result.loss,
                    self.config.selection_metric,
                    current_metric,
                )
            )
            self._emit_epoch_log(
                "epoch=%s nids_metrics | %s"
                % (
                    epoch,
                    self._format_nids_metrics(val_result.metrics),
                )
            )
            if self._show_tqdm() and hasattr(epoch_iterator, "set_postfix"):
                epoch_iterator.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_result.loss:.4f}",
                    metric=f"{current_metric:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            if early_stopping is not None and early_stopping.should_stop:
                self._emit_epoch_log(f"Early stopping at epoch {epoch}")
                break

        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

        summary = TrainingSummary(
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_model_path=str(best_model_path),
            history=history,
        )
        save_json(asdict(summary), self.output_dir / "training_summary.json")
        return summary

    def measure_latency(self, model: torch.nn.Module, data_loader: DataLoader) -> dict:
        device = next(model.parameters()).device
        return measure_inference_latency(model, data_loader, device=device)
