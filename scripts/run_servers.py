import signal
import subprocess
import sys
import time
from pathlib import Path

# ANSI colors
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"  # No Color

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Store process handles
processes = []


def print_colored(message, color):
    """Print a colored message."""
    print(f"{color}{message}{NC}")


def cleanup(signum=None, frame=None):
    """Clean up running processes."""
    print_colored("\nShutting down servers...", YELLOW)
    for proc in processes:
        if proc.poll() is None:  # If process is still running
            try:
                # Send SIGTERM
                proc.terminate()
                # Wait for a short time
                for _ in range(5):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                # If still running, force kill
                if proc.poll() is None:
                    proc.kill()
            except Exception as e:
                print(f"Error shutting down process: {e}")

    print_colored("All servers stopped.", GREEN)
    sys.exit(0)


def main():
    """Main function to start both servers."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Header
    print_colored("========================================", GREEN)
    print_colored("  Parkinson's Disease Detection Servers ", GREEN)
    print_colored("========================================", GREEN)

    # Start backend server
    print_colored("\nStarting FastAPI backend server...", YELLOW)
    backend_process = subprocess.Popen(["python", "api.py"], cwd=BACKEND_DIR, text=True)
    processes.append(backend_process)
    print_colored("Backend server running on http://localhost:8000", GREEN)
    print_colored("API Documentation: http://localhost:8000/docs", YELLOW)

    # Start frontend server
    print_colored("\nStarting React frontend server...", YELLOW)
    frontend_process = subprocess.Popen(["npm", "run", "dev"], cwd=FRONTEND_DIR, text=True)
    processes.append(frontend_process)
    print_colored("Frontend server running on http://localhost:5173", GREEN)

    print_colored("\nBoth servers are now running. Press Ctrl+C to stop.", YELLOW)

    # Monitor processes
    while True:
        # Check if any process has exited
        for proc in processes:
            if proc.poll() is not None:
                print_colored("\nOne of the servers has unexpectedly stopped.", RED)
                cleanup()
        time.sleep(1)


if __name__ == "__main__":
    main()
