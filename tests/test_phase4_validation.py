"""Phase 4 validation tests for staged distillation framework.

No torch/Isaac Lab runtime required.

Checks:
  1. train.py has staged orchestration helpers and stage pipeline path
  2. runner accepts stage_context and emits stage summary
  3. Tienkung/Walker agent cfg expose staged_training with amp/distill flags
  4. Tienkung/Walker flat env cfg define StageDistill env classes
    5. Tienkung/Walker gym registration includes StageDistill task IDs
    6. AMP/Distill utility module exists with core APIs
"""

import ast
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _banner(title: str):
    print(f"\n{BOLD}{'=' * 70}\n{title}\n{'=' * 70}{RESET}")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _ok(msg: str) -> bool:
    print(f"{GREEN}✓{RESET} {msg}")
    return True


def _fail(msg: str) -> bool:
    print(f"{RED}✗{RESET} {msg}")
    return False


def check_file(path: str, label: str) -> bool:
    return _ok(label) if os.path.isfile(path) else _fail(label)


def check_syntax(path: str, label: str) -> bool:
    try:
        ast.parse(_read(path))
        return _ok(f"Syntax valid: {label}")
    except SyntaxError as e:
        return _fail(f"Syntax error in {label}: line {e.lineno}: {e.msg}")


def check_in(src: str, needle: str, label: str) -> bool:
    return _ok(label) if needle in src else _fail(label)


def main() -> int:
    print(f"\n{BOLD}{'=' * 70}\nPHASE 4 VALIDATION TESTS\n{'=' * 70}{RESET}")
    results: list[bool] = []

    train_py = os.path.join(BASE, "scripts/rsl_rl/train.py")
    runner_py = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py")
    tk_agent = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/tienkung/agents/rsl_rl_ppo_cfg.py")
    wk_agent = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/walker/agents/rsl_rl_ppo_cfg.py")
    tk_env = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/tienkung/flat_env_cfg.py")
    wk_env = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/walker/flat_env_cfg.py")
    tk_reg = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/tienkung/__init__.py")
    wk_reg = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/walker/__init__.py")
    aux_py = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/utils/ppo_amp_distill.py")

    _banner("1. Files Exist")
    for p, label in [
        (train_py, "train.py exists"),
        (runner_py, "my_on_policy_runner.py exists"),
        (tk_agent, "tienkung agent cfg exists"),
        (wk_agent, "walker agent cfg exists"),
        (tk_env, "tienkung flat env cfg exists"),
        (wk_env, "walker flat env cfg exists"),
        (tk_reg, "tienkung registry exists"),
        (wk_reg, "walker registry exists"),
        (aux_py, "ppo_amp_distill.py exists"),
    ]:
        results.append(check_file(p, label))

    _banner("2. Syntax")
    for p, label in [
        (train_py, "train.py"),
        (runner_py, "my_on_policy_runner.py"),
        (tk_agent, "tienkung rsl_rl_ppo_cfg.py"),
        (wk_agent, "walker rsl_rl_ppo_cfg.py"),
        (tk_env, "tienkung flat_env_cfg.py"),
        (wk_env, "walker flat_env_cfg.py"),
        (tk_reg, "tienkung __init__.py"),
        (wk_reg, "walker __init__.py"),
        (aux_py, "ppo_amp_distill.py"),
    ]:
        results.append(check_syntax(p, label))

    train_src = _read(train_py)
    runner_src = _read(runner_py)
    tk_agent_src = _read(tk_agent)
    wk_agent_src = _read(wk_agent)
    tk_env_src = _read(tk_env)
    wk_env_src = _read(wk_env)
    tk_reg_src = _read(tk_reg)
    wk_reg_src = _read(wk_reg)
    aux_src = _read(aux_py)

    _banner("3. Staged Orchestration")
    results.append(check_in(train_src, "def _run_one_stage(", "train.py has _run_one_stage helper"))
    results.append(check_in(train_src, "staged_cfg = _as_dict", "train.py loads staged_training config"))
    results.append(check_in(train_src, "[STAGED] Running two-stage training pipeline", "train.py has staged execution path"))
    results.append(check_in(train_src, "teacher_source", "train.py supports teacher source strategy"))

    _banner("4. Runner Stage Context")
    results.append(check_in(runner_src, "stage_context", "runner accepts stage_context"))
    results.append(check_in(runner_src, "def emit_stage_summary", "runner can emit stage summary"))

    _banner("5. Agent Staged Config")
    results.append(check_in(tk_agent_src, "staged_training", "tienkung agent has staged_training"))
    results.append(check_in(tk_agent_src, "enable_amp", "tienkung stage config has enable_amp"))
    results.append(check_in(tk_agent_src, "enable_distill", "tienkung stage config has enable_distill"))
    results.append(check_in(wk_agent_src, "staged_training", "walker agent has staged_training"))

    _banner("6. Env & Task Registration")
    results.append(check_in(tk_env_src, "class TienkungFlatStageDistillEnvCfg", "tienkung stage env class exists"))
    results.append(check_in(wk_env_src, "class WalkerFlatStageDistillEnvCfg", "walker stage env class exists"))
    results.append(check_in(tk_reg_src, "Tracking-Flat-Tienkung-StageDistill-v0", "tienkung StageDistill task registered"))
    results.append(check_in(wk_reg_src, "Tracking-Flat-Walker-StageDistill-v0", "walker StageDistill task registered"))

    _banner("7. AMP/Distill Utility")
    results.append(check_in(aux_src, "class AMPDiscriminator", "AMPDiscriminator defined"))
    results.append(check_in(aux_src, "def amp_discriminator_loss(", "amp_discriminator_loss() defined"))
    results.append(check_in(aux_src, "def action_distill_loss(", "action_distill_loss() defined"))
    results.append(check_in(aux_src, "def feature_distill_loss(", "feature_distill_loss() defined"))

    _banner("VALIDATION SUMMARY")
    passed = sum(results)
    total = len(results)
    print(f"{BOLD}Total Tests:{RESET} {total}")
    print(f"{GREEN}Passed:{RESET} {passed}")
    print(f"{RED}Failed:{RESET} {total - passed}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ALL PHASE 4 VALIDATION TESTS PASSED!{RESET}\n")
        return 0

    failed = [i + 1 for i, r in enumerate(results) if not r]
    print(f"\n{RED}{BOLD}✗ {total - passed} tests failed (indices: {failed}){RESET}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
