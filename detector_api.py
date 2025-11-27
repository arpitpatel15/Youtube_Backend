import json, re, math, os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# YouTube API imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Please install the YouTube API client:")
    print("pip install 1")
    exit(1)

try:
    from isodate import parse_duration
except ImportError:
    print("Please install isodate for duration parsing:")
    print("pip install isodate")
    exit(1)

from sklearn.model_selection import RepeatedStratifiedKFold, GroupKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, average_precision_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
from scipy.stats import entropy
import joblib
from datetime import datetime
import unicodedata
from urllib.parse import urlparse, parse_qs

# Install textstat if needed: pip install textstat
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError:
    print("textstat not found. Install with: pip install textstat")
    # Fallback functions
    def flesch_reading_ease(text):
        return 0
    def flesch_kincaid_grade(text):
        return 0

# ========= ENHANCED USER CONFIG =========
API_KEY = "AIzaSyCZ3_29H0OqdqkVQUZdBLMBfztg-KB_yR4"  # Your YouTube API key
MODEL_PATH = "enhanced_fake_video_detector.joblib"
TEXT_MAX_WORD_FEATURES = 20000
TEXT_MAX_CHAR_FEATURES = 12000
RANDOM_STATE = 42
REAL_THRESHOLD = 0.5
USE_ENSEMBLE = True
ENABLE_ADVANCED_FEATURES = True
# ===============================

# Enhanced negative marker seeds with more comprehensive patterns
NEG_WORDS_SEED = [
    # English fake/clickbait indicators
    "fake", "hoax", "clickbait", "waste", "report", "scam", "buffering", "not working",
    "time waste", "boring", "stupid", "useless", "misleading", "lie", "lies", "fraud",
    "exposed", "truth revealed", "shocking", "unbelievable", "gone wrong", "prank",
    "you won't believe", "doctors hate", "secret revealed", "leaked", "scandal",

    # Hindi/Hinglish fake indicators
    "bakwaas", "bekar", "jhooth", "time kharab", "report karo", "report kar do",
    "explain nahi", "explain nhi", "nahi mila", "nhi mila", "movie nahi", "movie nhi",
    "ullu", "bewakoof", "faltu", "ghatiya", "kharab", "dhokha", "jhootha",

    # Engagement manipulation patterns
    "like subscribe", "smash like", "hit the bell", "notification squad", "first comment",
    "pin this comment", "heart this", "make this viral", "share if you", "repost if"
]
NEG_WORDS_SEED = list(dict.fromkeys([w.lower() for w in NEG_WORDS_SEED]))

# Positive indicators for real content
POS_WORDS_SEED = [
    "tutorial", "guide", "how to", "step by step", "explanation", "analysis", "review",
    "educational", "informative", "documentary", "news", "official", "verified",
    "research", "study", "facts", "data", "statistics", "evidence", "source",
    "interview", "expert", "professional", "academic", "scientific", "journal"
]
POS_WORDS_SEED = list(dict.fromkeys([w.lower() for w in POS_WORDS_SEED]))

# ---------- YouTube API Functions ----------

def get_video_id_from_url(url: str) -> str:
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            return query.get('v', [''])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/embed/')[-1]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')

    return ''

def get_youtube_client():
    """Initialize YouTube API client"""
    return build('youtube', 'v3', developerKey=API_KEY)

def get_channel_subscribers(youtube, channel_id: str) -> int:
    """Get channel subscriber count"""
    try:
        channel_response = youtube.channels().list(
            part="statistics",
            id=channel_id
        ).execute()

        if channel_response["items"]:
            stats = channel_response["items"][0]["statistics"]
            return int(stats.get("subscriberCount", 0))
    except:
        pass
    return 0

def get_video_comments(youtube, video_id: str, max_comments: int = 20) -> Tuple[List[str], List[str]]:
    """Get top comments and replies for a video"""
    comments = []
    replies = []

    try:
        comment_response = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=max_comments,
            order="relevance"
        ).execute()

        for item in comment_response["items"]:
            # Top-level comment
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)

            # Replies
            if "replies" in item:
                for reply in item["replies"]["comments"][:5]:  # Max 5 replies per comment
                    reply_text = reply["snippet"]["textDisplay"]
                    replies.append(reply_text)

    except Exception as e:
        print(f"Warning: Could not fetch comments: {e}")

    return comments, replies

def get_video_metadata_from_url(url: str) -> Dict[str, Any]:
    """Get complete video metadata from YouTube URL"""
    video_id = get_video_id_from_url(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    youtube = get_youtube_client()

    try:
        # Get video details
        video_response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()

        if not video_response["items"]:
            raise ValueError(f"No video found for ID {video_id}")

        item = video_response["items"][0]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        content_details = item["contentDetails"]

        # Parse duration
        duration_sec = parse_duration(content_details.get("duration", "PT0S")).total_seconds()

        # Get channel subscriber count
        channel_id = snippet.get("channelId", "")
        subscribers = get_channel_subscribers(youtube, channel_id)

        # Get comments
        comments, replies = get_video_comments(youtube, video_id)

        video_meta = {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": channel_id,
            "published_at": snippet.get("publishedAt", ""),
            "likes": int(stats.get("likeCount", 0)),
            "views": int(stats.get("viewCount", 0)),
            "subscribers": subscribers,
            "video_duration_seconds": int(duration_sec),
            "category": snippet.get("categoryId", ""),
            "upload_language": snippet.get("defaultAudioLanguage", "en"),
            "top_comments": [{"text": comment} for comment in comments],
            "top_replies": [{"text": reply} for reply in replies],
            "video_url": url,
            "is_real": None  # Unknown for prediction
        }

        return video_meta

    except HttpError as e:
        raise ValueError(f"YouTube API error: {e}")
    except Exception as e:
        raise ValueError(f"Error fetching video metadata: {e}")

# ---------- Enhanced Utility Functions (Same as before) ----------

def clean_int(x):
    """Enhanced number cleaning with better handling of formatted numbers"""
    if x is None or x == '':
        return 0
    s = str(x).lower().replace(',', '').replace(' ', '').strip()

    # Handle K, M, B suffixes
    multiplier = 1
    if 'k' in s:
        multiplier = 1000
        s = s.replace('k', '')
    elif 'm' in s:
        multiplier = 1000000
        s = s.replace('m', '')
    elif 'b' in s:
        multiplier = 1000000000
        s = s.replace('b', '')

    # Extract digits and decimal points
    digits = re.findall(r'\d+\.?\d*', s)
    if digits:
        try:
            return int(float(digits[0]) * multiplier)
        except:
            return 0
    return 0

def normalize_text(text: str) -> str:
    """Enhanced text normalization"""
    if not text:
        return ""

    # Unicode normalization
    try:
        text = unicodedata.normalize('NFKD', text)
    except:
        pass

    # Remove excessive punctuation and normalize whitespace
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def extract_advanced_text_features(text: str) -> Dict[str, float]:
    """Extract advanced linguistic features from text"""
    if not text or len(text.strip()) < 10:
        return {
            'caps_ratio': 0, 'exclamation_ratio': 0, 'question_ratio': 0,
            'reading_ease': 0, 'reading_grade': 0, 'word_diversity': 0,
            'avg_word_length': 0, 'sentence_count': 0, 'emoji_count': 0
        }

    # Basic ratios
    caps_ratio = len([c for c in text if c.isupper()]) / len(text) if text else 0
    exclamation_ratio = text.count('!') / len(text) if text else 0
    question_ratio = text.count('?') / len(text) if text else 0

    # Readability scores
    try:
        reading_ease = flesch_reading_ease(text)
        reading_grade = flesch_kincaid_grade(text)
    except:
        reading_ease = reading_grade = 0

    # Word diversity
    words = text.lower().split()
    word_diversity = len(set(words)) / len(words) if words else 0

    # Average word length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    # Sentence count
    sentence_count = len(re.findall(r'[.!?]+', text))

    # Emoji count
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))

    return {
        'caps_ratio': caps_ratio,
        'exclamation_ratio': exclamation_ratio,
        'question_ratio': question_ratio,
        'reading_ease': reading_ease,
        'reading_grade': reading_grade,
        'word_diversity': word_diversity,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'emoji_count': emoji_count
    }

def extract_engagement_features(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract advanced engagement and metadata features"""
    likes = clean_int(record.get("likes", 0))
    views = clean_int(record.get("views", 0))
    subscribers = clean_int(record.get("subscribers", 0))
    duration = int(record.get("video_duration_seconds", 0))

    # Advanced engagement ratios
    engagement_rate = (likes) / (views + 1)
    subscriber_view_ratio = subscribers / (views + 1)
    view_per_sub = views / (subscribers + 1)

    # Duration features
    is_short_video = int(duration < 60)
    is_long_video = int(duration > 1800)
    duration_category = 0 if duration < 300 else 1 if duration < 1200 else 2

    # Channel maturity
    channel_maturity = 0 if subscribers < 1000 else 1 if subscribers < 100000 else 2

    return {
        'engagement_rate': engagement_rate,
        'subscriber_view_ratio': subscriber_view_ratio,
        'view_per_sub': view_per_sub,
        'is_short_video': is_short_video,
        'is_long_video': is_long_video,
        'duration_category': duration_category,
        'channel_maturity': channel_maturity,
        'log_likes': np.log1p(likes),
        'log_views': np.log1p(views),
        'log_subscribers': np.log1p(subscribers),
        'log_duration': np.log1p(duration)
    }

def concat_texts(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Enhanced text concatenation with normalization"""
    title = normalize_text(record.get("title", "") or "")
    description = normalize_text(record.get("description", "") or "")

    comments = []
    for k in ["top_comments", "top_replies"]:
        for c in (record.get(k) or []):
            if isinstance(c, dict):
                text = normalize_text(c.get("text", ""))
            else:
                text = normalize_text(str(c))
            if text:
                comments.append(text)

    comments_text = " \n ".join(comments)
    combined = " \n ".join([title, description, comments_text])

    return title, description, comments_text, combined

def has_pattern(s: str, pat: str) -> int:
    """Enhanced pattern matching with normalization"""
    if not s:
        return 0
    s = normalize_text(s)
    return int(bool(re.search(pat, s, flags=re.IGNORECASE)))

def count_occurrences(text: str, phrases: List[str]) -> int:
    """Enhanced phrase counting with better matching"""
    if not text:
        return 0
    text = normalize_text(text.lower())
    count = 0
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        count += len(re.findall(pattern, text))
    return count

def extract_spam_features(text: str) -> Dict[str, int]:
    """Extract spam indicators from text"""
    if not text:
        return {'repeated_chars': 0, 'excessive_caps': 0, 'spam_urls': 0, 'phone_numbers': 0}

    repeated_chars = len(re.findall(r'(.)\1{3,}', text))

    words = text.split()
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    excessive_caps = int(caps_words > len(words) * 0.3)

    spam_urls = len(re.findall(r'bit\.ly|tinyurl|short\.link|free\.com', text, re.I))
    phone_numbers = len(re.findall(r'\b\d{10,}\b', text))

    return {
        'repeated_chars': repeated_chars,
        'excessive_caps': excessive_caps,
        'spam_urls': spam_urls,
        'phone_numbers': phone_numbers
    }

def build_enhanced_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build enhanced dataframe with advanced features"""
    rows = []

    for r in records:
        title, description, comments, combined = concat_texts(r)

        # Basic features
        likes = clean_int(r.get("likes"))
        views = clean_int(r.get("views"))
        subscribers = clean_int(r.get("subscribers"))
        duration = int(r.get("video_duration_seconds") or 0)

        total_comments = len((r.get("top_comments") or [])) + len((r.get("top_replies") or []))

        # Sentiment indicators
        neg_count = count_occurrences(comments, NEG_WORDS_SEED)
        pos_count = count_occurrences(comments, POS_WORDS_SEED)

        # Advanced text features
        title_features = extract_advanced_text_features(title)
        desc_features = extract_advanced_text_features(description)
        comment_features = extract_advanced_text_features(comments)

        # Engagement features
        engagement_features = extract_engagement_features(r)

        # Spam features
        spam_features = extract_spam_features(combined)

        # Content type patterns
        content_patterns = {
            'has_tutorial': has_pattern(title + " " + description, r'\b(tutorial|how\s*to|guide|step\s*by\s*step)\b'),
            'has_review': has_pattern(title + " " + description, r'\b(review|rating|opinion|thoughts)\b'),
            'has_news': has_pattern(title + " " + description, r'\b(news|breaking|latest|update|report)\b'),
            'has_entertainment': has_pattern(title + " " + description, r'\b(funny|comedy|meme|viral|trending)\b'),
            'has_educational': has_pattern(title + " " + description, r'\b(learn|education|explain|facts|science)\b'),
            'has_clickbait': has_pattern(title, r'\b(shocking|amazing|unbelievable|you\s*won\'?t\s*believe|secret|exposed)\b'),
            'has_full_movie': has_pattern(title, r'\b(full\s*movie|complete\s*film|entire\s*movie|फुल\s*मूवी)\b'),
            'has_fair_use': has_pattern(description, r'(fair\s*use|copyright\s*disclaimer|section\s*107|educational\s*purpose)'),
            'has_urgency': has_pattern(title, r'\b(urgent|hurry|limited\s*time|act\s*now|don\'?t\s*miss)\b')
        }

        # Build the row
        row = {
            # Identifiers
            'video_id': r.get("video_id", ""),
            'channel_id': r.get("channel_id", ""),

            # Text content
            'title': title,
            'description': description,
            'comments_text': comments,
            'combined_text': combined,

            # Basic length features
            'title_len': len(title),
            'desc_len': len(description),
            'comments_len': len(comments),
            'combined_len': len(combined),

            # Basic engagement
            'likes': likes,
            'views': views,
            'subscribers': subscribers,
            'duration': duration,
            'total_comments': total_comments,

            # Ratios
            'like_ratio': likes / (views + 1),
            'comment_ratio': total_comments / (views + 1),

            # Sentiment
            'neg_comment_count': neg_count,
            'pos_comment_count': pos_count,
            'neg_comment_ratio': neg_count / (total_comments + 1),
            'pos_comment_ratio': pos_count / (total_comments + 1),
            'sentiment_balance': (pos_count - neg_count) / (pos_count + neg_count + 1),

            # Target
            'label': int(bool(r.get("is_real"))) if r.get("is_real") is not None else 0
        }

        # Add advanced features if enabled
        if ENABLE_ADVANCED_FEATURES:
            # Text features
            for prefix, features in [('title', title_features), ('desc', desc_features), ('comment', comment_features)]:
                for key, value in features.items():
                    row[f'{prefix}_{key}'] = value

            # Engagement features
            row.update(engagement_features)

            # Spam features
            row.update(spam_features)

            # Content patterns
            row.update(content_patterns)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def enhanced_explain_prediction(pipe: Pipeline, sample_df_row: pd.Series,
                              real_threshold: float = REAL_THRESHOLD, k: int = 8) -> Dict[str, Any]:
    """Enhanced prediction explanation with better interpretability"""
    try:
        # Get prediction
        sample_df = pd.DataFrame([sample_df_row])
        X_sample = sample_df.drop(columns=["label"], errors='ignore')

        prob_real = pipe.predict_proba(X_sample)[0, 1]
        pred_label = int(prob_real >= real_threshold)
        confidence = prob_real if pred_label == 1 else 1 - prob_real

        # Extract key information for explanation
        title = sample_df_row.get('title', '')
        views = sample_df_row.get('views', 0)
        likes = sample_df_row.get('likes', 0)
        subscribers = sample_df_row.get('subscribers', 0)
        neg_comments = sample_df_row.get('neg_comment_count', 0)
        pos_comments = sample_df_row.get('pos_comment_count', 0)

        # Build explanation
        if pred_label == 1:
            verdict = "REAL"
            explanation = "This video appears to be authentic based on"
            if ENABLE_ADVANCED_FEATURES:
                factors = []
                if sample_df_row.get('has_educational', 0):
                    factors.append("educational content patterns")
                if sample_df_row.get('has_tutorial', 0):
                    factors.append("tutorial/instructional format")
                if sample_df_row.get('pos_comment_ratio', 0) > 0.1:
                    factors.append("positive audience engagement")
                if sample_df_row.get('engagement_rate', 0) > 0.01:
                    factors.append("healthy engagement metrics")
                if not factors:
                    factors = ["content quality indicators", "engagement patterns"]
                explanation += " " + ", ".join(factors[:3])
        else:
            verdict = "FAKE/MISLEADING"
            explanation = "This video shows signs of being fake/misleading due to"
            if ENABLE_ADVANCED_FEATURES:
                factors = []
                if sample_df_row.get('has_clickbait', 0):
                    factors.append("clickbait title patterns")
                if sample_df_row.get('has_full_movie', 0):
                    factors.append("full movie claim (likely copyright violation)")
                if sample_df_row.get('neg_comment_ratio', 0) > 0.1:
                    factors.append("negative audience feedback")
                if sample_df_row.get('spam_urls', 0) > 0:
                    factors.append("spam links in content")
                if not factors:
                    factors = ["suspicious content patterns", "engagement anomalies"]
                explanation += " " + ", ".join(factors[:3])

        # Key metrics for evidence
        evidence = [
            f"Confidence Score: {confidence:.3f}",
            f"Engagement Rate: {likes/(views+1):.4f}",
            f"Subscriber-View Ratio: {subscribers/(views+1):.4f}"
        ]

        if neg_comments > 0 or pos_comments > 0:
            evidence.append(f"Comment Sentiment: {pos_comments} positive, {neg_comments} negative")

        return {
            "verdict": verdict,
            "confidence_score": round(confidence, 4),
            "probability_real": round(prob_real, 4),
            "reasoning": explanation,
            "key_evidence": evidence[:k],
            "raw_features": {
                "title": title[:100] + "..." if len(title) > 100 else title,
                "views": int(views),
                "likes": int(likes),
                "subscribers": int(subscribers),
                "engagement_rate": round(likes/(views+1), 6),
                "duration": sample_df_row.get('duration', 0)
            }
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "probability_real": 0.0,
            "reasoning": f"Prediction explanation failed: {str(e)}",
            "key_evidence": [],
            "raw_features": {}
        }

# ---------- Main Prediction Function ----------

def predict_video_from_url(url: str, model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """
    Main function to predict if a YouTube video is fake or real from URL

    Args:
        url: YouTube video URL
        model_path: Path to trained model

    Returns:
        Dictionary with prediction results
    """
    try:
        print(f"🎬 Analyzing YouTube video...")
        print(f"📹 URL: {url}")

        # Step 1: Extract video metadata from YouTube API
        print("\n🔍 Fetching video metadata from YouTube API...")
        video_metadata = get_video_metadata_from_url(url)

        print(f"✓ Video Title: {video_metadata['title'][:80]}...")
        print(f"✓ Channel: {video_metadata['channel_title']}")
        print(f"✓ Views: {video_metadata['views']:,}")
        print(f"✓ Likes: {video_metadata['likes']:,}")
        print(f"✓ Subscribers: {video_metadata['subscribers']:,}")
        print(f"✓ Comments fetched: {len(video_metadata['top_comments'])}")

        # Step 2: Load trained model
        print(f"\n🤖 Loading trained model from {model_path}...")
        try:
            pipe = joblib.load(model_path)
            print("✓ Model loaded successfully!")
        except FileNotFoundError:
            return {
                "verdict": "ERROR",
                "confidence_score": 0.0,
                "reasoning": f"Model file not found: {model_path}. Please train the model first.",
                "key_evidence": [],
                "raw_features": {}
            }

        # Step 3: Process data for prediction
        print("\n⚙️ Processing video data...")
        df = build_enhanced_dataframe([video_metadata])

        # Remove label column for prediction
        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        # Step 4: Make prediction
        print("\n🔮 Making prediction...")
        sample_row = df.iloc[0]
        result = enhanced_explain_prediction(pipe, sample_row)

        # Add video metadata to result
        result["video_metadata"] = {
            "title": video_metadata["title"],
            "channel": video_metadata["channel_title"],
            "views": video_metadata["views"],
            "likes": video_metadata["likes"],
            "duration": video_metadata["video_duration_seconds"],
            "published": video_metadata["published_at"]
        }

        return result

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "probability_real": 0.0,
            "reasoning": f"Analysis failed: {str(e)}",
            "key_evidence": [],
            "raw_features": {},
            "video_metadata": {}
        }

# ---------- Interactive Interface ----------

def main():
    print("=" * 80)
    print("🎯 YOUTUBE FAKE VIDEO DETECTOR")
    print("=" * 80)
    print("This system analyzes YouTube videos to predict if they are REAL or FAKE")
    print("using machine learning and video metadata analysis.")
    print("-" * 80)

    while True:
        try:
            # Get URL from user
            print("\n📝 Enter a YouTube video URL (or 'quit' to exit):")
            url = input("🔗 URL: ").strip()

            if url.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using YouTube Fake Video Detector!")
                break

            if not url:
                print("❌ Please enter a valid URL")
                continue

            # Validate URL
            if 'youtube.com' not in url and 'youtu.be' not in url:
                print("❌ Please enter a valid YouTube URL")
                continue

            print("\n" + "=" * 80)

            # Analyze video
            result = predict_video_from_url(url)

            # Display results
            print("\n" + "🎯 PREDICTION RESULTS")
            print("=" * 80)

            if result["verdict"] == "ERROR":
                print(f"❌ {result['reasoning']}")
            else:
                # Display verdict with colors/emojis
                if result["verdict"] == "REAL":
                    print(f"✅ VERDICT: {result['verdict']}")
                    print(f"🎯 Confidence: {result['confidence_score']:.1%}")
                else:
                    print(f"⚠️  VERDICT: {result['verdict']}")
                    print(f"🎯 Confidence: {result['confidence_score']:.1%}")

                print(f"📊 Probability of being REAL: {result['probability_real']:.1%}")
                print(f"🤔 Reasoning: {result['reasoning']}")

                print("\n🔍 Key Evidence:")
                for i, evidence in enumerate(result['key_evidence'], 1):
                    print(f"  {i}. {evidence}")

                if result.get('video_metadata'):
                    meta = result['video_metadata']
                    print(f"\n📹 Video Information:")
                    print(f"  📺 Title: {meta.get('title', 'N/A')[:60]}...")
                    print(f"  🎪 Channel: {meta.get('channel', 'N/A')}")
                    print(f"  👀 Views: {meta.get('views', 0):,}")
                    print(f"  👍 Likes: {meta.get('likes', 0):,}")
                    print(f"  ⏱️  Duration: {meta.get('duration', 0)} seconds")

            print("=" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different URL.")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using the training script.")
        print("The model should be saved as 'enhanced_fake_video_detector.joblib'")
        exit(1)

    main()
