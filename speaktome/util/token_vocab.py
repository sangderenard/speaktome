"""Token vocabulary helpers for simple encode/decode mappings."""
# --- END HEADER ---
class TokenVocabulary:
    """Simple token vocabulary wrapper.

    Provides bidirectional mapping between tokens and ids. Allows future
    extensions for embedding strategies and orientation-aware scoring.
    """

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, text):
        return [self.token_to_id[c] for c in text if c in self.token_to_id]

    def decode(self, ids):
        return ''.join(self.id_to_token.get(i, '?') for i in ids)

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def test() -> None:
        """Run a basic encode/decode self-check."""
        vocab = TokenVocabulary("ab ")
        ids = vocab.encode("ba")
        assert vocab.decode(ids) == "ba"
