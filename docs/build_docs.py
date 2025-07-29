#!/usr/bin/env python3
"""
Sphinx documentation build script for JEPA framework.

This script provides utilities for building, serving, and managing
the documentation for the JEPA project.
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = Path(__file__).parent
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "build"
HTML_DIR = BUILD_DIR / "html"


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
        
    return result


def install_requirements():
    """Install documentation requirements."""
    print("Installing documentation requirements...")
    
    requirements_file = DOCS_DIR / "requirements.txt"
    if not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}")
        return False
        
    run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    print("Documentation requirements installed successfully!")
    return True


def clean_build():
    """Clean the build directory."""
    print("Cleaning build directory...")
    
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"Removed {BUILD_DIR}")
    else:
        print("Build directory doesn't exist, nothing to clean")


def build_docs(clean=False, verbose=False, builder="html"):
    """Build the documentation."""
    if clean:
        clean_build()
        
    print(f"Building documentation with {builder} builder...")
    
    # Create build directory if it doesn't exist
    BUILD_DIR.mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        "sphinx-build",
        "-b", builder,
        str(SOURCE_DIR),
        str(BUILD_DIR / builder)
    ]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")  # Quiet mode, but don't treat warnings as errors for now
        
    # Run sphinx-build
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print(f"Documentation built successfully!")
        print(f"Output: {BUILD_DIR / builder}")
        if builder == "html":
            print(f"Open: {BUILD_DIR / builder / 'index.html'}")
    else:
        print("Documentation build failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
        
    return True


def serve_docs(port=8000, host="127.0.0.1"):
    """Serve the documentation locally."""
    if not HTML_DIR.exists():
        print("HTML documentation not found. Building first...")
        if not build_docs():
            return False
            
    print(f"Serving documentation at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        import http.server
        import socketserver
        
        os.chdir(HTML_DIR)
        handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer((host, port), handler) as httpd:
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nStopping documentation server...")
    except ImportError:
        # Fallback to python -m http.server
        cmd = [sys.executable, "-m", "http.server", str(port), "--bind", host]
        run_command(cmd, cwd=HTML_DIR, check=False)


def watch_docs():
    """Watch for changes and rebuild documentation automatically."""
    try:
        import watchdog
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("watchdog package required for watch mode")
        print("Install with: pip install watchdog")
        return False
        
    class DocHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(('.rst', '.md', '.py')):
                print(f"File changed: {event.src_path}")
                build_docs(verbose=False)
                
    # Initial build
    build_docs()
    
    # Setup file watcher
    event_handler = DocHandler()
    observer = Observer()
    observer.schedule(event_handler, str(SOURCE_DIR), recursive=True)
    observer.schedule(event_handler, str(PROJECT_ROOT), recursive=True)
    
    observer.start()
    print(f"Watching for changes in {SOURCE_DIR} and {PROJECT_ROOT}")
    print("Press Ctrl+C to stop watching")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching for changes")
        
    observer.join()


def check_links():
    """Check for broken links in the documentation."""
    print("Checking for broken links...")
    
    result = run_command([
        "sphinx-build",
        "-b", "linkcheck",
        str(SOURCE_DIR),
        str(BUILD_DIR / "linkcheck")
    ], check=False)
    
    if result.returncode == 0:
        print("All links are valid!")
    else:
        print("Found broken links. Check the linkcheck output.")
        
    return result.returncode == 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="JEPA Documentation Builder")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install documentation requirements")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build documentation")
    build_parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    build_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    build_parser.add_argument("--builder", "-b", default="html", help="Sphinx builder to use")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve documentation locally")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve on")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to serve on")
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for changes and rebuild")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean build directory")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check for broken links")
    
    args = parser.parse_args()
    
    if args.command == "install":
        install_requirements()
    elif args.command == "build":
        build_docs(clean=args.clean, verbose=args.verbose, builder=args.builder)
    elif args.command == "serve":
        serve_docs(port=args.port, host=args.host)
    elif args.command == "watch":
        watch_docs()
    elif args.command == "clean":
        clean_build()
    elif args.command == "check":
        check_links()
    else:
        # Default: build and serve
        if build_docs():
            serve_docs()


if __name__ == "__main__":
    main()
