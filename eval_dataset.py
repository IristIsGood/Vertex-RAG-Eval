# eval_dataset.py
# Each item includes a "key_phrase" — a phrase that should appear in the
# correct retrieved chunk. Used for Recall@K and MRR metrics.

EVAL_DATASET = [
    {
        "question": "What is the central thesis of the book?",
        "ground_truth": (
            "The author's central thesis is that successful trading requires "
            "changing how we think — especially how we think about losing. "
            "The mindset of accepting and managing losses well is what separates "
            "winning traders from losing ones."
        ),
        "key_phrase": "losing",
    },
    {
        "question": "Why does the author say normal thinking does not work in trading?",
        "ground_truth": (
            "Because the human mind is hard-coded to avoid pain and keep us alive. "
            "This survival instinct works against trading, where accepting losses "
            "and acting against natural emotional reactions is required to succeed."
        ),
        "key_phrase": "pain",
    },
    {
        "question": "What is the role of emotion in trading according to the author?",
        "ground_truth": (
            "The author argues that emotions kill trading accounts. Successful "
            "traders must learn to desensitize their emotional response to fear, "
            "greed, and other natural reactions in order to act in their own best "
            "interest under pressure."
        ),
        "key_phrase": "emotion",
    },
    {
        "question": "What is the recipe for chocolate cake?",
        "ground_truth": "Not available in this document.",
        "key_phrase": None,  # not in document — should not match anything
    },
    {
        "question": "Who is the author of this book?",
        "ground_truth": "Tom Hougaard.",
        "key_phrase": "Hougaard",
    },
]