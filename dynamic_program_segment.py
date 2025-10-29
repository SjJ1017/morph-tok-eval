def segment(word: str, vocab: dict[str, float]) -> list[str]:
    """Segment a word into subwords using dynamic programming as in Unigram LM.

    It finds a segmentation that maximizes the sum of scores of the subwords. If
    the scores are log probabilities, it finds the most probable segmentation.
    If there is no way to build a segmentation from the given vocabulary, it
    uses characters as a fallback, with a -1000 penalty for each character.

    Args:
        word (str): The input word to segment.
        vocab (dict[str, float]): The vocabulary with scores (e.g., log probabilities) for each subword.

    Returns:
        list[str]: The segmented subwords.
    """
    costs = [0. for _ in range(len(word) + 1)]
    prev = [0 for _ in word]

    # First, dynamic programming
    for i in range(1, len(word) + 1):
        scores = []
        indices = []
        for j in range(i):
            subword_candidate = word[j:i]
            if subword_candidate in vocab:
                new_cost = costs[j] + vocab[subword_candidate]
                scores.append(new_cost)
                indices.append(j)
        if not scores:
            costs[i] = -1000
            prev[i - 1] = i - 1
        else:
            idx = max(range(len(scores)), key=lambda i: scores[i])

            costs[i] = scores[idx]
            prev[i - 1] = indices[idx]

    # Second, reconstrct the best options
    subwords = []
    idx = len(prev) - 1
    while idx >= 0:
        new_idx = prev[idx]
        sbwrd = word[new_idx:idx + 1]
        subwords.append(sbwrd)

        idx = new_idx - 1
    return list(reversed(subwords)), costs[-1]