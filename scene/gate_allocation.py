import torch
import torch.nn as nn


class GateAllocator(nn.Module):
    def __init__(self, input_dim):
        super(GateAllocator, self).__init__()
        self.input_dim = int(input_dim)
        self.gate_head = nn.Linear(self.input_dim, 1)

    @staticmethod
    def _temperature(iteration, init, final, max_steps):
        init = float(init)
        final = float(final)
        max_steps = max(1, int(max_steps or 1))
        if iteration is None:
            return init
        progress = max(0.0, min(float(iteration) / float(max_steps), 1.0))
        return init + (final - init) * progress

    @staticmethod
    def _topk_indices(logits, detail_budget):
        detail_budget = max(0, min(int(detail_budget), logits.numel()))
        if detail_budget <= 0:
            return torch.empty(0, device=logits.device, dtype=torch.long)
        return torch.topk(logits, detail_budget, sorted=False).indices

    def select(
            self,
            gate_feature,
            detail_budget,
            training=False,
            train_mode="soft_all",
            eval_mode="topk",
            iteration=None,
            temperature_init=1.0,
            temperature_final=0.2,
            temperature_max_steps=30000,
            budget_lambda=0.01,
            binary_lambda=0.001,
    ):
        logits = self.gate_head(gate_feature).squeeze(-1)
        temperature = self._temperature(
            iteration,
            temperature_init,
            temperature_final,
            temperature_max_steps,
        )
        temperature = max(float(temperature), 1e-6)
        soft_gate = torch.sigmoid(logits / temperature)

        train_mode = str(train_mode or "soft_all").lower()
        eval_mode = str(eval_mode or "topk").lower()
        use_soft_all = training and train_mode == "soft_all"
        use_topk = (not training and eval_mode == "topk") or (training and not use_soft_all)

        if use_soft_all:
            selected_idx = torch.arange(logits.numel(), device=logits.device, dtype=torch.long)
        elif use_topk:
            selected_idx = self._topk_indices(logits, detail_budget)
        else:
            selected_idx = self._topk_indices(logits, detail_budget)

        selected_gate = soft_gate[selected_idx] if selected_idx.numel() > 0 else soft_gate.new_empty(0)
        loss = soft_gate.new_zeros(())
        budget_loss = soft_gate.new_zeros(())
        binary_loss = soft_gate.new_zeros(())
        budget_weight = float(budget_lambda)
        binary_weight = float(binary_lambda)
        if training and detail_budget > 0:
            if budget_weight != 0.0:
                budget = soft_gate.new_tensor(float(detail_budget))
                budget_loss = ((soft_gate.sum() - budget) / torch.clamp_min(budget, 1.0)).pow(2)
            if binary_weight != 0.0:
                binary_loss = (soft_gate * (1.0 - soft_gate)).mean()
            loss = budget_weight * budget_loss + binary_weight * binary_loss

        stats = {
            "gate_temperature": float(temperature),
            "gate_mass": float(soft_gate.detach().sum().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_mean": float(soft_gate.detach().mean().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_max": float(soft_gate.detach().max().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_min": float(soft_gate.detach().min().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_budget_loss": float(budget_loss.detach().item()),
            "gate_binary_loss": float(binary_loss.detach().item()),
        }
        return {
            "logits": logits,
            "soft_gate": soft_gate,
            "selected_idx": selected_idx,
            "selected_gate": selected_gate,
            "losses": {
                "loss": loss,
                "budget": budget_loss,
                "binary": binary_loss,
            },
            "stats": stats,
        }


class SlotGateSTAllocator(nn.Module):
    def __init__(self, input_dim):
        super(SlotGateSTAllocator, self).__init__()
        self.input_dim = int(input_dim)
        self.gate_head = nn.Linear(self.input_dim, 1)

    @staticmethod
    def _temperature(iteration, init, final, max_steps):
        return GateAllocator._temperature(iteration, init, final, max_steps)

    @staticmethod
    def _topk_indices(logits, detail_budget):
        return GateAllocator._topk_indices(logits, detail_budget)

    def select(
            self,
            gate_feature,
            detail_budget,
            training=False,
            train_mode="topk_st",
            eval_mode="topk",
            iteration=None,
            temperature_init=1.0,
            temperature_final=0.2,
            temperature_max_steps=30000,
            budget_lambda=0.0,
            binary_lambda=0.0,
    ):
        logits = self.gate_head(gate_feature).squeeze(-1)
        detail_budget = max(0, min(int(detail_budget), logits.numel()))
        temperature = self._temperature(
            iteration,
            temperature_init,
            temperature_final,
            temperature_max_steps,
        )
        temperature = max(float(temperature), 1e-6)

        if logits.numel() == 0 or detail_budget <= 0:
            soft_gate = logits.new_zeros(logits.shape)
            selected_idx = torch.empty(0, device=logits.device, dtype=torch.long)
            selected_gate = logits.new_empty(0)
            loss = logits.new_zeros(())
            stats = {
                "gate_temperature": float(temperature),
                "gate_mass": 0.0,
                "gate_mean": 0.0,
                "gate_max": 0.0,
                "gate_min": 0.0,
                "gate_budget_loss": 0.0,
                "gate_binary_loss": 0.0,
            }
            return {
                "logits": logits,
                "soft_gate": soft_gate,
                "selected_idx": selected_idx,
                "selected_gate": selected_gate,
                "losses": {
                    "loss": loss,
                    "budget": loss,
                    "binary": loss,
                },
                "stats": stats,
            }

        soft_gate = float(detail_budget) * torch.softmax(logits / temperature, dim=0)
        selected_idx = self._topk_indices(logits, detail_budget)
        selected_soft = soft_gate[selected_idx]
        if training:
            selected_hard = torch.ones_like(selected_soft)
            selected_gate = selected_hard - selected_soft.detach() + selected_soft
        else:
            selected_gate = torch.ones_like(selected_soft)

        loss = soft_gate.new_zeros(())
        budget_loss = soft_gate.new_zeros(())
        binary_loss = soft_gate.new_zeros(())
        budget_weight = float(budget_lambda)
        binary_weight = float(binary_lambda)
        if training:
            if budget_weight != 0.0:
                budget = soft_gate.new_tensor(float(detail_budget))
                budget_loss = ((soft_gate.sum() - budget) / torch.clamp_min(budget, 1.0)).pow(2)
            if binary_weight != 0.0:
                bounded_gate = torch.clamp(soft_gate, 0.0, 1.0)
                binary_loss = (bounded_gate * (1.0 - bounded_gate)).mean()
            loss = budget_weight * budget_loss + binary_weight * binary_loss

        stats = {
            "gate_temperature": float(temperature),
            "gate_mass": float(soft_gate.detach().sum().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_mean": float(soft_gate.detach().mean().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_max": float(soft_gate.detach().max().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_min": float(soft_gate.detach().min().item()) if soft_gate.numel() > 0 else 0.0,
            "gate_budget_loss": float(budget_loss.detach().item()),
            "gate_binary_loss": float(binary_loss.detach().item()),
        }
        return {
            "logits": logits,
            "soft_gate": soft_gate,
            "selected_idx": selected_idx,
            "selected_gate": selected_gate,
            "losses": {
                "loss": loss,
                "budget": budget_loss,
                "binary": binary_loss,
            },
            "stats": stats,
        }

