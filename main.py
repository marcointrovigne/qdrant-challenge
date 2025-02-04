import argparse
import sys

from loguru import logger

from qdrant_logic import QdrantDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage Qdrant dataset operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--action",
        required=True,
        choices=["create", "upload", "search", "create_and_upload"],
        help="Action to perform"
    )

    parser.add_argument(
        "--query",
        help="Search query text"
    )
    parser.add_argument(
        "--user-id",
        help="Filter results by user ID"
    )

    parser.add_argument(
        "--force-delete",
        action="store_true",
        help="Force delete existing collection if it exists"
    )

    return parser.parse_args()

def handle_create(dataset: QdrantDataset, args) -> None:
    """Handle collection creation."""
    dataset.create_collection(force_delete=args.force_delete)
    logger.info("Collection created successfully")


def handle_upload(dataset: QdrantDataset, args) -> None:
    """Handle data upload."""
    dataset.upload_batch()
    logger.info("Data upload completed")


def handle_search(dataset: QdrantDataset, args) -> None:
    """Handle search operation."""
    if not args.query:
        logger.error("Search query is required for search action")
        sys.exit(1)

    logger.info(f"Performing search with query: {args.query}")
    results = dataset.hybrid_search(
        query=args.query,
        user_id=args.user_id
    )

    # Display search results
    logger.info(f"Found {len(results)} results:")
    for idx, point in enumerate(results, 1):
        logger.info(f"   {idx}. Score: {point.score:.4f}")
        logger.info(f"   Title: {point.payload['text']}")
        logger.info(f"   User ID: {point.payload['user_id']}\n")

def main():
    """Main entry point for the script."""
    args = parse_args()

    try:
        # Initialize dataset with configuration
        dataset = QdrantDataset()

        # Handle different actions
        if args.action == "create":
            handle_create(dataset, args)
        elif args.action == "upload":
            handle_upload(dataset, args)
        elif args.action == "search":
            handle_search(dataset, args)
        elif args.action == "create_and_upload":
            handle_create(dataset, args)
            handle_upload(dataset, args)

    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()