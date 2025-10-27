"""
Command-line interface for DeepProstate
"""
import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DeepProstate - AI-powered prostate MRI analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepprostate              # Launch GUI (default)
  deepprostate --gui        # Launch GUI explicitly
  deepprostate --version    # Show version information
        """
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch graphical user interface"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )

    args = parser.parse_args()

    if args.version:
        from deepprostate import __version__
        print(f"DeepProstate version {__version__}")
        return 0

    if args.gui or len(sys.argv) == 1:
        # Launch GUI
        try:
            from deepprostate.application import DeepProstateApplication
            app = DeepProstateApplication()
            return app.run()
        except ImportError as e:
            print(f"Error: Cannot launch GUI: {e}")
            print("Make sure all dependencies are installed: pip install deepprostate")
            return 1
        except Exception as e:
            print(f"Error launching GUI: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
