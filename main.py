# main.py

import argparse
from pipeline.prediction_pipeline import hybrid_recommendation


def main():
    parser = argparse.ArgumentParser(
        description="Anime Recommender CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    sub = parser.add_subparsers(dest="command", help="Available Commands")

    # -----------------------------
    # ✅ Command: recommend
    # -----------------------------
    recommend_cmd = sub.add_parser("recommend", help="Get recommendations for a user")
    recommend_cmd.add_argument(
        "--user_id",
        type=int,
        required=True,
        help="User ID for generating anime recommendations"
    )
    recommend_cmd.add_argument(
        "--user_weight",
        type=float,
        default=0.5,
        help="Weight for collaborative filtering"
    )
    recommend_cmd.add_argument(
        "--content_weight",
        type=float,
        default=0.5,
        help="Weight for content-based filtering"
    )
    recommend_cmd.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of recommendations to return"
    )

    # -----------------------------
    # ✅ Command: test
    # -----------------------------
    sub.add_parser("test", help="Simple system test")

    # -----------------------------
    # ✅ Command: version
    # -----------------------------
    sub.add_parser("version", help="Show CLI version")

    args = parser.parse_args()

    # -----------------------------
    # ✅ CLI Logic
    # -----------------------------
    if args.command == "recommend":
        print(f"\n🎯 Fetching recommendations for User {args.user_id}...\n")
        results = hybrid_recommendation(
            user_id=args.user_id,
            user_weight=args.user_weight,
            content_weight=args.content_weight,
            top_k=args.top_k
        )
        print("\n✅ Final Recommendations:")
        for idx, anime in enumerate(results, 1):
            print(f"{idx}. {anime}")

    elif args.command == "test":
        print("✅ System is working! CLI test successful.")

    elif args.command == "version":
        print("Anime Recommender CLI v1.0.0")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()