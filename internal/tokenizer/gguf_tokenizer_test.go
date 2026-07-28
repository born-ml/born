package tokenizer

import "testing"

func TestGGUFTokenizer_GPT2ByteLevel(t *testing.T) {
	// Minimal vocab with individual chars + merged tokens.
	tokens := []string{
		"!", `"`, "#", "$", "%", "&", "'", "(", ")", "*",
		"H", "e", "l", "o", "w", "r", "d",
		"Ġ", // byte-level space
		"He", "ll", "ello",
		"Ġw", "Ġworld",
		"Hello", "world",
	}
	merges := []string{
		"H e",     // → He
		"l l",     // → ll
		"He llo",  // → Hello (won't match, no "llo")
		"Ġ w",     // → Ġw
		"Ġw orld", // → Ġworld (won't match, no "orld")
	}

	tok, err := NewGGUFTokenizer(tokens, merges, "gpt2", -1, 0, -1, -1)
	if err != nil {
		t.Fatal(err)
	}

	// "Hello world" → pre-tokenize → ["Hello", " world"]
	// byte-level: ["Hello", "Ġworld"]
	// BPE on "Hello": H+e→He, l+l→ll, then He+ll→? no merge. Chars: He, ll, o
	// BPE on "Ġworld": Ġ+w→Ġw, then Ġw+orld→? no merge.
	ids, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Fatal("expected non-empty token IDs")
	}

	// Decode should recover the original text (byte-level decode)
	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", text)
	}
}

func TestGGUFTokenizer_SentencePiece(t *testing.T) {
	tokens := []string{
		"<s>", "</s>", "▁Hello", "▁world", "Hello", "world",
	}

	tok, err := NewGGUFTokenizer(tokens, nil, "llama", 0, 1, -1, -1)
	if err != nil {
		t.Fatal(err)
	}

	ids, err := tok.Encode("Hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Fatal("expected non-empty token IDs")
	}

	text, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	// SentencePiece uses ▁ for spaces; Decode converts them back and strips
	// the leading sentence-start marker.
	expected := "Hello world"
	if text != expected {
		t.Errorf("expected %q, got %q", expected, text)
	}
}

func TestGPT2ByteToUnicode_RoundTrip(t *testing.T) {
	// Every byte should round-trip through byte-level encoding.
	for b := 0; b < 256; b++ {
		encoded := byteLevelEncode(string([]byte{byte(b)}))
		decoded := byteLevelDecode(encoded)
		if decoded != string([]byte{byte(b)}) {
			t.Errorf("byte %d: round-trip failed, got %q", b, decoded)
		}
	}
}
