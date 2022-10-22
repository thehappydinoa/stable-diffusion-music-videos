import argparse

from .app import demo


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the app on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the app in debug mode",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the app publicly",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    demo.launch(server_port=args.port, debug=args.debug, share=args.share)


if __name__ == "__main__":
    main()
