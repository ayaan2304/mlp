from transformers import pipeline

# Load sentiment analysis model
sentiment_classifier = pipeline("sentiment-analysis")

def detect_sentiment(text, neutral_threshold=0.55):
    """
    Detects sentiment of the input text.
    Returns: (sentiment_label, confidence_score)
    """
    res = sentiment_classifier(text)[0]
    label = res['label'].lower()  # 'positive' or 'negative'
    score = res['score']
    # Treat low-confidence cases as neutral
    if score < neutral_threshold:
        return "neutral", score
    return label, score


def check_alignment(generated_text, target_sentiment):
    """
    Verifies if generated text sentiment matches the target sentiment.
    Returns: (is_aligned, predicted_label, confidence_score)
    """
    detected_label, score = detect_sentiment(generated_text)
    is_aligned = detected_label == target_sentiment
    return is_aligned, detected_label, score
