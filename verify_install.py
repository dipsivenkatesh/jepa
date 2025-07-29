#!/usr/bin/env python3
"""
Installation verification script for JEPA package.

This script verifies that JEPA has been installed correctly and all
components are importable.
"""

import sys
import traceback
from typing import List, Tuple


def check_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✅ {module_name} {description}".strip()
    except ImportError as e:
        return False, f"❌ {module_name} {description}: {e}".strip()
    except Exception as e:
        return False, f"❌ {module_name} {description}: Unexpected error - {e}".strip()


def main():
    """Main verification function."""
    print("🔍 JEPA Installation Verification")
    print("=" * 40)
    
    # Core imports to check
    imports_to_check = [
        ("jepa", "- Main package"),
        ("jepa.models", "- Model components"),
        ("jepa.models.jepa", "- JEPA model"),
        ("jepa.trainer", "- Training framework"),
        ("jepa.trainer.trainer", "- Main trainer"),
        ("jepa.config", "- Configuration system"),
        ("jepa.data", "- Data utilities"),
        ("jepa.loggers", "- Logging system"),
        ("jepa.cli", "- Command line interface"),
    ]
    
    # Check imports
    results: List[Tuple[bool, str]] = []
    for module, desc in imports_to_check:
        success, message = check_import(module, desc)
        results.append((success, message))
        print(message)
    
    print("\n" + "=" * 40)
    
    # Check if main classes can be imported
    print("🔧 Testing Key Components")
    print("-" * 40)
    
    try:
        from jepa import JEPA, JEPATrainer, load_config
        print("✅ Core classes imported successfully")
        
        # Test version access
        import jepa
        print(f"✅ JEPA version: {jepa.__version__}")
        
    except Exception as e:
        print(f"❌ Failed to import core classes: {e}")
        traceback.print_exc()
        return False
    
    # Check CLI commands
    print("\n🖥️  CLI Commands Available")
    print("-" * 40)
    
    try:
        from jepa.cli import train_main, evaluate_main
        print("✅ CLI commands available")
    except Exception as e:
        print(f"❌ CLI commands not available: {e}")
    
    # Summary
    failed_imports = [msg for success, msg in results if not success]
    
    print("\n" + "=" * 40)
    if failed_imports:
        print("❌ Installation Issues Found:")
        for msg in failed_imports:
            print(f"   {msg}")
        print(f"\n{len(failed_imports)} issues found out of {len(results)} checks.")
        return False
    else:
        print("✅ All components installed successfully!")
        print("\n🎉 JEPA is ready to use!")
        print("\nNext steps:")
        print("  1. Try: python -c 'import jepa; print(jepa.__version__)'")
        print("  2. Run: jepa-train --help")
        print("  3. Check examples: python examples/usage_example.py")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
