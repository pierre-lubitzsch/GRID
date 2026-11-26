"""CPU tests for src/eval_mia.py's attack pieces. Run: python -m pytest -q test_eval_mia.py"""

import torch

from src.eval_mia import auc, fit_logreg, mia_aucs


def test_auc_matches_closed_form_with_ties():
    assert auc(torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0])) == 1.0
    assert auc(torch.tensor([0.0, 1.0, 2.0, 3.0]), torch.tensor([1, 1, 0, 0])) == 0.0
    # every score tied -> 0.5
    assert auc(torch.ones(6), torch.tensor([1, 1, 1, 0, 0, 0])) == 0.5
    # pos {2, 1}, neg {1, 0}: pairs (2>1),(2>0),(1=1 -> .5),(1>0) = 3.5 / 4
    assert abs(auc(torch.tensor([2.0, 1.0, 1.0, 0.0]), torch.tensor([1, 1, 0, 0])) - 0.875) < 1e-12


def test_logreg_separates_and_mia_reads_forgetting():
    g = torch.Generator().manual_seed(0)
    members = torch.randn(400, 4, generator=g) - 1.0       # low loss = member-like
    test = torch.randn(400, 4, generator=g) + 1.0
    s = fit_logreg(torch.cat([members, test]), torch.cat([torch.ones(400), torch.zeros(400)]))
    assert auc(torch.cat([s(members), s(test)]), torch.cat([torch.ones(400), torch.zeros(400)])) > 0.9
    remembered = torch.randn(300, 4, generator=g) - 1.0     # forget set still member-like
    forgotten = torch.randn(300, 4, generator=g) + 1.0      # forget set looks like test
    assert mia_aucs(members, test, remembered)["auc_forget_vs_test"] > 0.9
    assert abs(mia_aucs(members, test, forgotten)["auc_forget_vs_test"] - 0.5) < 0.08
