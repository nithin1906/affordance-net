import argparse
import subprocess
import sys
import os
from pathlib import Path

# Get the absolute path of the project root directory.
# The script is in the root, so its parent is the root.
ROOT_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"

def run_server(host="127.0.0.1", port=8000):
    """
    Starts both the backend and frontend servers.
    """
    # --- Backend (Uvicorn) ---
    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.server:app",
        f"--host={host}",
        f"--port={port}",
        "--reload",
    ]
    
    # --- Frontend (Vite) ---
    # Use shell=True on Windows for npm commands
    frontend_command = "npm run dev"

    print(f"🚀 Starting backend server on http://{host}:{port}")
    print(f"🚀 Starting frontend server (likely on http://localhost:5173)")
    print("----------------------------------------------------")
    print("Press Ctrl+C to stop both servers.")

    backend_process = None
    frontend_process = None

    try:
        backend_process = subprocess.Popen(
            backend_command,
            cwd=ROOT_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        frontend_process = subprocess.Popen(
            frontend_command,
            cwd=FRONTEND_DIR,
            shell=True,  # Necessary for 'npm' on Windows
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Wait for both processes to complete
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received. Shutting down servers...")
    finally:
        if backend_process and backend_process.poll() is None:
            print("Terminating backend process...")
            backend_process.terminate()
            backend_process.wait()

        if frontend_process and frontend_process.poll() is None:
            print("Terminating frontend process...")
            frontend_process.terminate()
            frontend_process.wait()
        
        print("✅ Servers have been shut down.")


def main():
    """
    Main function to parse arguments and run commands.
    """
    parser = argparse.ArgumentParser(
        description="A management script for the AffordanceNet project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- runserver command ---
    parser_runserver = subparsers.add_parser(
        "runserver", help="Run the backend and frontend development servers."
    )
    parser_runserver.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the backend server."
    )
    parser_runserver.add_argument(
        "--port", type=int, default=8000, help="Port for the backend server."
    )
    parser_runserver.set_defaults(func=lambda args: run_server(args.host, args.port))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
