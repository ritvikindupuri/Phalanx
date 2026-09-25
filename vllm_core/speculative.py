"""
Speculative Decoding Engine
Implements speculative execution with Draft + Target verification sampling
(Leviathan et al. / Chen et al.), generating multiple tokens per target forward pass.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class SpeculativeVerifier:
    """
    Mathematical Speculative Verification Engine.
    Verifies draft tokens in parallel using rejection sampling with exact distribution matching.
    """
    def __init__(self, gamma: int = 4):
        """
        :param gamma: Number of draft tokens generated speculatively per step.
        """
        self.gamma = gamma
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.target_forward_passes = 0

    def verify_tokens(
        self,
        draft_tokens: List[int],
        draft_probs: List[np.ndarray],   # Shape: (gamma, vocab_size)
        target_probs: List[np.ndarray],  # Shape: (gamma + 1, vocab_size)
        temperature: float = 1.0
    ) -> Tuple[List[int], int]:
        """
        Verifies draft tokens against target model distributions.
        Returns:
            accepted_tokens: The verified tokens (including 1 corrective/bonus token).
            num_accepted_draft: How many draft tokens were accepted before the first rejection.
        """
        accepted: List[int] = []
        num_accepted = 0
        self.target_forward_passes += 1

        for i, token in enumerate(draft_tokens):
            self.total_draft_tokens += 1
            p_draft = draft_probs[i][token]
            p_target = target_probs[i][token]

            # Rejection sampling acceptance condition:
            # accept with probability min(1, p_target / p_draft)
            accept_prob = min(1.0, (p_target / max(1e-9, p_draft)))
            rand_val = np.random.uniform(0.0, 1.0)

            if rand_val <= accept_prob:
                accepted.append(token)
                num_accepted += 1
                self.accepted_draft_tokens += 1
            else:
                # Token rejected! Sample corrective token from adjusted residual distribution
                res_dist = np.maximum(0.0, target_probs[i] - draft_probs[i])
                sum_res = np.sum(res_dist)
                if sum_res > 1e-9:
                    res_dist /= sum_res
                    corrective_token = int(np.random.choice(len(res_dist), p=res_dist))
                else:
                    corrective_token = int(np.argmax(target_probs[i]))
                
                accepted.append(corrective_token)
                return accepted, num_accepted

        # If all gamma tokens were accepted, sample one bonus token from target distribution at (gamma)
        if len(target_probs) > len(draft_tokens):
            bonus_probs = target_probs[len(draft_tokens)]
            bonus_token = int(np.random.choice(len(bonus_probs), p=bonus_probs))
            accepted.append(bonus_token)

        return accepted, num_accepted

    def get_stats(self) -> Dict[str, Any]:
        acceptance_rate = (
            (self.accepted_draft_tokens / self.total_draft_tokens * 100.0)
            if self.total_draft_tokens > 0 else 0.0
        )
        avg_tokens_per_pass = (
            (self.accepted_draft_tokens + self.target_forward_passes) / self.target_forward_passes
            if self.target_forward_passes > 0 else 1.0
        )
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "acceptance_rate_pct": round(acceptance_rate, 2),
            "target_forward_passes": self.target_forward_passes,
            "speedup_ratio": round(avg_tokens_per_pass, 2)
        }
