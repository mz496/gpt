

from src.tiny_shakespeare.tokenizer.byte_pair_encoding import BytePairEncoding


def test_encode_decode_roundtrip() -> None:
    bpe = BytePairEncoding()
    s = "aaabdaaabac"
    bpe.encode(s)
    print(bpe)

def test_encode_decode_roundtrip_long() -> None:
    bpe = BytePairEncoding()
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    bpe.encode(text)

    