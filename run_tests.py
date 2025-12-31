import pytest
import sys
import os

def run_all_tests():
    """
    Simulates running 'pytest tests' to execute all tests in the tests/ directory.
    """
    print("🚀 Starting all tests...")
    
    # Arguments for pytest:
    # 'tests' : the directory to search for tests
    # '-v'    : verbose output
    # '-x'    : stop on first failure (optional, can be removed)
    pytest_args = ["tests", "-v"]
    
    # Run pytest programmatically
    retcode = pytest.main(pytest_args)
    
    if retcode == 0:
        print("\n✅ All tests passed successfully! Ready to push.")
        sys.exit(0)
    else:
        print(f"\n❌ Tests failed with exit code: {retcode}. Please fix errors before pushing.")
        sys.exit(retcode)

if __name__ == "__main__":
    run_all_tests()
