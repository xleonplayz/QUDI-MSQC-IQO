#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for the NV-Zentrum Quantum Computer Simulator.

This script runs all tests for the simulator and generates a coverage report.
It can be run with:
    python run_tests.py

Additional options:
    --core-only: Only run core simulator tests
    --integration-only: Only run integration tests
    --verbose: Show detailed test output
    --coverage: Generate coverage report

Copyright (c) 2023, the qudi developers.
"""

import os
import sys
import unittest
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_tests(core_only=False, integration_only=False, verbose=False):
    """Run the test suite.
    
    Args:
        core_only: If True, only run core simulator tests
        integration_only: If True, only run integration tests
        verbose: If True, show detailed test output
    
    Returns:
        test_result: Result of test run
    """
    # Determine which test modules to import
    if core_only:
        from test_simulator_core import TestNVSimulatorCore
        test_suite = unittest.TestLoader().loadTestsFromTestCase(TestNVSimulatorCore)
    elif integration_only:
        try:
            from test_simulator_integration import run_simulator_test_suite
            return run_simulator_test_suite()
        except ImportError as e:
            logger.error(f"Error importing integration tests: {e}")
            sys.exit(1)
    else:
        # Import all test modules
        test_suite = unittest.defaultTestLoader.discover(os.path.dirname(__file__), 
                                                        pattern="test_*.py")
    
    # Run the tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    test_result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Ran {test_result.testsRun} tests")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Skipped: {len(test_result.skipped)}")
    
    return test_result


def run_with_coverage(core_only=False, integration_only=False, verbose=False):
    """Run tests with coverage reporting.
    
    Args:
        core_only: If True, only run core simulator tests
        integration_only: If True, only run integration tests
        verbose: If True, show detailed test output
    
    Returns:
        test_result: Result of test run
    """
    try:
        import coverage
    except ImportError:
        logger.error("Coverage package not found. Install with 'pip install coverage'.")
        sys.exit(1)
    
    # Start coverage
    cov = coverage.Coverage(source=['nv_simulator'])
    cov.start()
    
    # Run tests
    result = run_tests(core_only, integration_only, verbose)
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report
    cov.html_report(directory=os.path.join(os.path.dirname(__file__), 'htmlcov'))
    print(f"HTML coverage report saved to: {os.path.join(os.path.dirname(__file__), 'htmlcov/index.html')}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NV-Zentrum Simulator tests')
    parser.add_argument('--core-only', action='store_true', help='Only run core simulator tests')
    parser.add_argument('--integration-only', action='store_true', help='Only run integration tests')
    parser.add_argument('--verbose', action='store_true', help='Show detailed test output')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    # Check for mutually exclusive options
    if args.core_only and args.integration_only:
        logger.error("Cannot specify both --core-only and --integration-only")
        sys.exit(1)
    
    # Run tests
    if args.coverage:
        result = run_with_coverage(args.core_only, args.integration_only, args.verbose)
    else:
        result = run_tests(args.core_only, args.integration_only, args.verbose)
    
    # Exit with appropriate code (0 for success, 1 for failures)
    sys.exit(len(result.failures) + len(result.errors))