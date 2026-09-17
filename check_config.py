#!/usr/bin/env python3
"""Quick diagnostic script to check multi-agent-generator configuration."""

import os
from pathlib import Path


def check_env_file():
    """Check .env file for correct timeout values."""
    print("=" * 60)
    print("CHECKING .ENV CONFIGURATION")
    print("=" * 60)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        return
    
    with open(env_path) as f:
        content = f.read()
    
    checks = {
        "AGENT_EXECUTION_TIMEOUT=300": "Agent execution timeout",
        "TEST_EXECUTION_TIMEOUT=60": "Test execution timeout",
        "DEPENDENCY_INSTALL_TIMEOUT=1800": "Dependency install timeout",
    }
    
    for key, desc in checks.items():
        if key in content:
            print(f"✅ {desc}: {key.split('=')[1]}s")
        else:
            print(f"❌ {desc}: NOT FOUND or WRONG VALUE")
            print(f"   Expected: {key}")
    print()


def check_pytest_timeout():
    """Check pytest.ini timeout in generated projects."""
    print("=" * 60)
    print("CHECKING PYTEST TIMEOUT CONFIGURATION")
    print("=" * 60)
    
    files_to_check = [
        "multi_agent_generator/emit/shared.py",
        "multi_agent_generator/evaluation/test_generator.py",
    ]
    
    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            print(f"❌ {filepath}: NOT FOUND")
            continue
        
        with open(path) as f:
            content = f.read()
        
        if 'timeout = 10' in content:
            print(f"✅ {filepath}: timeout = 10")
        elif 'timeout = 60' in content:
            print(f"❌ {filepath}: Still has timeout = 60 (should be 10)")
        else:
            print(f"⚠️  {filepath}: timeout setting not found")
    print()


def check_deps_config():
    """Check dependency configuration."""
    print("=" * 60)
    print("CHECKING DEPENDENCY CONFIGURATION")
    print("=" * 60)
    
    deps_path = Path("multi_agent_generator/execution/deps.py")
    if not deps_path.exists():
        print("❌ deps.py not found!")
        return
    
    with open(deps_path) as f:
        content = f.read()
    
    checks = {
        '"--timeout",': "Pip timeout flag",
        '"30",': "Pip timeout value",
        '"--retries",': "Pip retries flag",
        '"2",': "Pip retries value",
    }
    
    for key, desc in checks.items():
        if key in content:
            print(f"✅ {desc}: Present")
        else:
            print(f"❌ {desc}: Missing")
    print()


def check_role_matching():
    """Check role matching logic."""
    print("=" * 60)
    print("CHECKING ROLE MATCHING LOGIC")
    print("=" * 60)
    
    req_spec_path = Path("multi_agent_generator/core/requirements_spec.py")
    if not req_spec_path.exists():
        print("❌ requirements_spec.py not found!")
        return
    
    with open(req_spec_path) as f:
        content = f.read()
    
    if "needle_words = set(needle.split())" in content:
        print("✅ Fuzzy word matching: Enabled")
    else:
        print("❌ Fuzzy word matching: Not implemented")
    
    if "any(word in a.label.lower() for word in needle_words)" in content:
        print("✅ Word-level agent matching: Enabled")
    else:
        print("❌ Word-level agent matching: Not implemented")
    print()


def check_crewai_version():
    """Check CrewAI version requirement."""
    print("=" * 60)
    print("CHECKING CREWAI VERSION")
    print("=" * 60)
    
    deps_path = Path("multi_agent_generator/dependencies.py")
    if not deps_path.exists():
        print("❌ dependencies.py not found!")
        return
    
    with open(deps_path) as f:
        content = f.read()
    
    if 'crewai>=0.70.0' in content:
        print("✅ CrewAI version: >=0.70.0 (relaxed)")
    elif 'crewai>=0.80.0' in content:
        print("⚠️  CrewAI version: >=0.80.0 (strict, may cause issues)")
    else:
        print("❌ CrewAI version requirement not found")
    print()


def check_pytest_command():
    """Check that pytest command has timeout override."""
    print("=" * 60)
    print("CHECKING PYTEST COMMAND LINE TIMEOUT")
    print("=" * 60)
    
    runner_path = Path("multi_agent_generator/execution/runner.py")
    if not runner_path.exists():
        print("❌ runner.py not found!")
        return
    
    with open(runner_path) as f:
        content = f.read()
    
    if '"--timeout=5"' in content:
        print("✅ Pytest command line: --timeout=5 (overrides pytest.ini)")
    elif '"--timeout' in content:
        print("⚠️  Pytest has timeout but value might be wrong")
    else:
        print("❌ No --timeout flag in pytest command (will use pytest.ini value)")
    print()


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MULTI-AGENT GENERATOR DIAGNOSTIC" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    check_env_file()
    check_pytest_timeout()
    check_pytest_command()
    check_deps_config()
    check_role_matching()
    check_crewai_version()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Review the checks above. All items should show ✅")
    print("If you see ❌ or ⚠️, the fixes may not have been applied correctly.")
    print()
    print("To apply fixes again, refer to FIXES_APPLIED.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
