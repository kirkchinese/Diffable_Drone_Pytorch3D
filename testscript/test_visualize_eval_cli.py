import ast
import re
import subprocess
import sys


def test_help_works_without_heavy_imports():
    """Verify that --help renders without importing heavy modules like pytorch3d.
    
    This test ensures that argparse help can render without triggering the module-level
    imports of DroneSimulator, Model, Model_bigger, SceneGenerator which depend on pytorch3d.
    """
    # Run --help in a subprocess to ensure clean import environment
    result = subprocess.run(
        [sys.executable, "visualize_eval.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Should succeed (exit code 0)
    assert result.returncode == 0, (
        f"visualize_eval.py --help failed with exit code {result.returncode}.\n"
        f"STDERR: {result.stderr}\n"
        f"STDOUT: {result.stdout}"
    )
    
    # Should contain expected CLI arguments in help output
    assert "--checkpoint" in result.stdout, "Help should mention --checkpoint"
    assert "--num_episodes" in result.stdout, "Help should mention --num_episodes"
    assert "--random_scene" in result.stdout, "Help should mention --random_scene"
    
    # Should NOT contain pytorch3d import errors
    assert "ModuleNotFoundError" not in result.stderr, (
        f"Help should not trigger ModuleNotFoundError.\nSTDERR: {result.stderr}"
    )
    assert "pytorch3d" not in result.stderr, (
        f"Help should not try to import pytorch3d.\nSTDERR: {result.stderr}"
    )


def test_new_cli_args_exist():
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    required_args = [
        "--goal_radius",
        "--stagnation_window",
        "--stagnation_progress",
        "--stagnation_speed",
        "--spin_near_goal_radius",
        "--spin_yaw_thresh",
    ]
    
    missing = []
    for arg in required_args:
        pattern = rf"add_argument\s*\(\s*['\"]({re.escape(arg)})['\"]"
        if not re.search(pattern, source):
            missing.append(arg)
    
    assert not missing, f"Missing CLI args in source: {missing}"


def test_classify_episode_is_used():
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    has_nav_import = any(
        (isinstance(n, ast.ImportFrom) and n.module == "testscript.navigation_metrics")
        for n in ast.walk(tree)
    )
    
    assert has_nav_import, "visualize_eval.py must import from testscript.navigation_metrics"


def test_no_duplicate_summary_block_in_print_summary():
    """Verify print_summary function does not contain duplicate summary code blocks."""
    with open("visualize_eval.py", "r") as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    print_summary_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "print_summary":
            print_summary_func = node
            break
    
    assert print_summary_func is not None, "Could not find print_summary function"
    
    loops_over_all_records = []
    for node in ast.walk(print_summary_func):
        if isinstance(node, ast.For):
            iter_node = node.iter
            if isinstance(iter_node, ast.Name) and iter_node.id == "all_records":
                loops_over_all_records.append(node)
            elif isinstance(iter_node, ast.Call):
                if isinstance(iter_node.func, ast.Name) and iter_node.func.id == "enumerate":
                    if iter_node.args and isinstance(iter_node.args[0], ast.Name):
                        if iter_node.args[0].id == "all_records":
                            loops_over_all_records.append(node)
    
    assert len(loops_over_all_records) == 1, (
        f"print_summary should have exactly 1 loop over all_records, "
        f"but found {len(loops_over_all_records)} - duplicate summary code block"
    )


def test_run_episode_does_not_overwrite_gru_hidden_state_variable():
    with open("visualize_eval.py", "r") as f:
        source = f.read()

    assert "act_raw, _, h = self.model(x, state, h)" in source
    assert "h = self.env.R[:, :, 0]" not in source, (
        "run_episode() overwrites the GRU hidden state variable `h` with heading data, "
        "which causes recurrent inference to fail on later timesteps. Use a different "
        "local name for heading vectors."
    )
