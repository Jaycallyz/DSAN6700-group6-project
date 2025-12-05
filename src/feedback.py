#!/usr/bin/env python
"""
Feedback Analysis Tool

Usage:
    python scripts/view_feedback.py
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_feedback(feedback_file="data/feedback/user_feedback.jsonl"):
    """upload all feedback file"""
    feedback_path = Path(feedback_file)
    
    if not feedback_path.exists():
        print(f"❌ No feedback file found at {feedback_file}")
        print("ℹ️  Users haven't submitted any feedback yet.")
        return []
    
    feedbacks = []
    with open(feedback_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                feedbacks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return feedbacks

def analyze_feedback(feedbacks):
    """analyze feedback"""
    if not feedbacks:
        print("No feedback to analyze.")
        return
    
    print("\n" + "="*60)
    print("📊 FEEDBACK ANALYSIS REPORT")
    print("="*60)
    
    # Total count
    print(f"\n📈 Total Feedback Submissions: {len(feedbacks)}")
    
    # Satisfaction distribution
    print("\n😊 Satisfaction Distribution:")
    satisfaction_counts = {}
    for fb in feedbacks:
        sat = fb['feedback']['satisfaction']
        satisfaction_counts[sat] = satisfaction_counts.get(sat, 0) + 1
    
    for sat, count in sorted(satisfaction_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(feedbacks)) * 100
        print(f"  {sat}: {count} ({percentage:.1f}%)")
    
    # Star ratings
    print("\n⭐ Star Ratings:")
    stars = [fb['feedback']['stars'] for fb in feedbacks]
    avg_stars = sum(stars) / len(stars)
    print(f"  Average: {avg_stars:.2f} / 5.0")
    print(f"  Distribution: {dict(pd.Series(stars).value_counts().sort_index())}")
    
    # Comments
    comments = [fb['feedback']['comments'] for fb in feedbacks if fb['feedback']['comments']]
    print(f"\n💬 Comments: {len(comments)} comments received")
    
    if comments:
        print("\n📝 Recent Comments:")
        for i, comment in enumerate(comments[-5:], 1):  # Last 5 comments
            print(f"  {i}. \"{comment}\"")
    
    # Popular inputs
    print("\n🎬 Most Common Movie Inputs:")
    movies = [fb['inputs']['movie'] for fb in feedbacks]
    movie_counts = pd.Series(movies).value_counts().head(5)
    for movie, count in movie_counts.items():
        print(f"  {movie}: {count} times")
    
    print("\n📚 Most Common Book Inputs:")
    books = [fb['inputs']['book'] for fb in feedbacks]
    book_counts = pd.Series(books).value_counts().head(5)
    for book, count in book_counts.items():
        print(f"  {book}: {count} times")
    
    print("\n🎵 Most Common Music Inputs:")
    music = [fb['inputs']['music'] for fb in feedbacks]
    music_counts = pd.Series(music).value_counts().head(5)
    for track, count in music_counts.items():
        print(f"  {track}: {count} times")
    
    # Top recommended destinations
    print("\n🌍 Most Frequently Recommended Destinations:")
    all_destinations = []
    for fb in feedbacks:
        for rec in fb['recommendations']:
            all_destinations.append(f"{rec['name']}, {rec['country']}")
    
    dest_counts = pd.Series(all_destinations).value_counts().head(10)
    for dest, count in dest_counts.items():
        print(f"  {dest}: {count} times")
    
    # Temporal analysis
    print("\n📅 Submission Timeline:")
    dates = [datetime.fromisoformat(fb['timestamp']).date() for fb in feedbacks]
    date_counts = pd.Series(dates).value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count} submissions")
    
    print("\n" + "="*60)

def export_to_csv(feedbacks, output_file="data/feedback/feedback_summary.csv"):
    """导出为 CSV 格式"""
    if not feedbacks:
        print("No feedback to export.")
        return
    
    # Flatten data
    rows = []
    for fb in feedbacks:
        row = {
            'timestamp': fb['timestamp'],
            'movie': fb['inputs']['movie'],
            'book': fb['inputs']['book'],
            'music': fb['inputs']['music'],
            'satisfaction': fb['feedback']['satisfaction'],
            'stars': fb['feedback']['stars'],
            'comments': fb['feedback']['comments'],
            'movie_found': '✅' in fb['search_status'].get('movie', ''),
            'book_found': '✅' in fb['search_status'].get('book', ''),
            'music_found': '✅' in fb['search_status'].get('music', ''),
            'top_destination': fb['recommendations'][0]['name'] if fb['recommendations'] else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Exported to {output_file}")

def main():
    feedbacks = load_feedback()
    
    if feedbacks:
        analyze_feedback(feedbacks)
        export_to_csv(feedbacks)
    else:
        print("\n💡 Tip: Run the Streamlit app and submit feedback to see analysis here!")

if __name__ == "__main__":
    main()