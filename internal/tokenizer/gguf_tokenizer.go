package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// GGUFTokenizer is a tokenizer built from GGUF metadata.
//
// It supports two GGUF tokenizer model types:
//   - "gpt2": byte-level BPE with explicit merge rules
//   - "llama"/"unigram": SentencePiece-style with ▁ space markers (greedy longest-match)
//
// For GPT-2 BPE, the tokens in the GGUF vocab are already in byte-level unicode
// representation (e.g. space → Ġ U+0120). The merges array provides BPE merge rules.
// For SentencePiece, tokens use ▁ (U+2581) as space marker and tokenization uses
// greedy longest-match with score-based ranking.
type GGUFTokenizer struct {
	vocab         map[string]int32
	reverseVocab  map[int32]string
	mergeRanks    map[string]int // "first second" -> rank (0 = highest priority)
	byteLevel     bool           // true for gpt2 model type
	bosToken      int32
	eosToken      int32
	padToken      int32
	unkToken      int32
	specialTokens map[int32]bool
	// specialTokenStrings maps special token strings (e.g. "<|im_start|>") to IDs.
	// These are looked up directly, bypassing BPE.
	specialTokenStrings map[string]int32
}

// NewGGUFTokenizer builds a tokenizer from GGUF-extracted metadata.
//
// Parameters:
//   - tokens: the tokenizer.ggml.tokens array (vocab strings)
//   - merges: the tokenizer.ggml.merges array (BPE merge rules, "first second" per entry)
//   - modelType: tokenizer.ggml.model ("gpt2", "llama", "unigram")
//   - bosID, eosID, padID, unkID: special token IDs (-1 if not applicable)
func NewGGUFTokenizer(
	tokens []string,
	merges []string,
	modelType string,
	bosID, eosID, padID, unkID int32,
) (*GGUFTokenizer, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("gguf tokenizer: empty token list")
	}

	vocab := make(map[string]int32, len(tokens))
	reverseVocab := make(map[int32]string, len(tokens))
	for i, tok := range tokens {
		vocab[tok] = int32(i)
		reverseVocab[int32(i)] = tok
	}

	mergeRanks := make(map[string]int, len(merges))
	for i, m := range merges {
		mergeRanks[m] = i
	}

	byteLevel := strings.EqualFold(modelType, "gpt2")

	specialTokens := make(map[int32]bool)
	specialTokenStrings := make(map[string]int32)
	for _, id := range []int32{bosID, eosID, padID, unkID} {
		if id >= 0 {
			specialTokens[id] = true
		}
	}

	// Scan vocab for special tokens (patterns like <|...|> or <s>, </s>, etc.).
	// These bypass BPE and are looked up directly.
	for tok, id := range vocab {
		if isSpecialTokenString(tok) {
			specialTokenStrings[tok] = id
			specialTokens[id] = true
		}
	}

	return &GGUFTokenizer{
		vocab:               vocab,
		reverseVocab:        reverseVocab,
		mergeRanks:          mergeRanks,
		byteLevel:           byteLevel,
		bosToken:            bosID,
		eosToken:            eosID,
		padToken:            padID,
		unkToken:            unkID,
		specialTokens:       specialTokens,
		specialTokenStrings: specialTokenStrings,
	}, nil
}

// Encode converts text to token IDs.
//
// Special tokens (e.g. <|im_start|>, <|im_end|>) are looked up directly,
// bypassing BPE. Regular text between special tokens is encoded normally.
func (t *GGUFTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return []int32{}, nil
	}

	// If there are no special token strings, fall back to direct BPE.
	if len(t.specialTokenStrings) == 0 {
		return t.encodeText(text)
	}

	// Split text on special token boundaries.
	segments := splitOnSpecialTokens(text, t.specialTokenStrings)

	var result []int32
	for _, seg := range segments {
		if seg.isSpecial {
			if id, ok := t.specialTokenStrings[seg.text]; ok {
				result = append(result, id)
			}
		} else if seg.text != "" {
			ids, err := t.encodeText(seg.text)
			if err != nil {
				return nil, err
			}
			result = append(result, ids...)
		}
	}
	return result, nil
}

// encodeText encodes regular text (no special tokens) using BPE or greedy match.
func (t *GGUFTokenizer) encodeText(text string) ([]int32, error) {

	var preTokens []string
	if t.byteLevel {
		preTokens = gpt2PreTokenize(text)
	} else {
		preTokens = sentencePiecePreTokenize(text)
	}

	var result []int32
	for _, word := range preTokens {
		var encoded string
		if t.byteLevel {
			encoded = byteLevelEncode(word)
		} else {
			encoded = word
		}

		if len(t.mergeRanks) > 0 {
			result = append(result, t.bpeEncode(encoded)...)
		} else {
			result = append(result, t.greedyEncode(encoded)...)
		}
	}

	return result, nil
}

// bpeEncode applies BPE merge rules to a pre-tokenized word.
func (t *GGUFTokenizer) bpeEncode(word string) []int32 {
	symbols := splitToChars(word)

	for len(symbols) > 1 {
		bestRank := -1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			rank, ok := t.mergeRanks[pair]
			if ok && (bestRank == -1 || rank < bestRank) {
				bestRank = rank
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	var ids []int32
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			ids = append(ids, id)
		} else if t.unkToken >= 0 {
			ids = append(ids, t.unkToken)
		}
	}
	return ids
}

// greedyEncode uses greedy longest-match when no merge rules are available.
func (t *GGUFTokenizer) greedyEncode(word string) []int32 {
	chars := []rune(word)
	var ids []int32

	for i := 0; i < len(chars); {
		found := false
		for j := len(chars); j > i; j-- {
			substr := string(chars[i:j])
			if id, ok := t.vocab[substr]; ok {
				ids = append(ids, id)
				i = j
				found = true
				break
			}
		}
		if !found {
			if t.unkToken >= 0 {
				ids = append(ids, t.unkToken)
			}
			i++
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *GGUFTokenizer) Decode(tokens []int32) (string, error) {
	var parts []string
	for _, tok := range tokens {
		if text, ok := t.reverseVocab[tok]; ok {
			parts = append(parts, text)
		}
	}

	joined := strings.Join(parts, "")

	if t.byteLevel {
		return byteLevelDecode(joined), nil
	}

	return joined, nil
}

func (t *GGUFTokenizer) VocabSize() int      { return len(t.vocab) }
func (t *GGUFTokenizer) BosToken() int32     { return t.bosToken }
func (t *GGUFTokenizer) EosToken() int32     { return t.eosToken }
func (t *GGUFTokenizer) PadToken() int32     { return t.padToken }
func (t *GGUFTokenizer) UnkToken() int32     { return t.unkToken }
func (t *GGUFTokenizer) IsSpecialToken(id int32) bool { return t.specialTokens[id] }

// --- GPT-2 byte-level encoding ---

var gpt2ByteToUnicode = func() map[byte]rune {
	m := make(map[byte]rune)
	// Printable ASCII ranges that map to themselves.
	for b := byte('!'); b <= byte('~'); b++ {
		m[b] = rune(b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		m[byte(b)] = rune(b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		m[byte(b)] = rune(b)
	}
	// Remaining bytes map to 256+n.
	n := 0
	for b := 0; b < 256; b++ {
		if _, ok := m[byte(b)]; !ok {
			m[byte(b)] = rune(256 + n)
			n++
		}
	}
	return m
}()

var gpt2UnicodeToByte = func() map[rune]byte {
	m := make(map[rune]byte, len(gpt2ByteToUnicode))
	for b, r := range gpt2ByteToUnicode {
		m[r] = b
	}
	return m
}()

func byteLevelEncode(text string) string {
	var sb strings.Builder
	sb.Grow(len(text))
	for _, b := range []byte(text) {
		sb.WriteRune(gpt2ByteToUnicode[b])
	}
	return sb.String()
}

func byteLevelDecode(text string) string {
	var buf []byte
	for _, r := range text {
		if b, ok := gpt2UnicodeToByte[r]; ok {
			buf = append(buf, b)
		}
	}
	return string(buf)
}

// --- Pre-tokenization ---

// gpt2PreTokenize splits text into pre-tokens for GPT-2 BPE.
// Each pre-token includes its leading space (if any).
func gpt2PreTokenize(text string) []string {
	var result []string
	runes := []rune(text)
	i := 0

	for i < len(runes) {
		if runes[i] == ' ' {
			j := i + 1
			if j < len(runes) {
				k := j
				if isLetterOrDigit(runes[j]) {
					for k < len(runes) && isLetterOrDigit(runes[k]) {
						k++
					}
				} else if isPunctuation(runes[j]) {
					for k < len(runes) && isPunctuation(runes[k]) {
						k++
					}
				} else {
					k = j + 1
				}
				result = append(result, string(runes[i:k]))
				i = k
			} else {
				result = append(result, " ")
				i++
			}
		} else if isLetterOrDigit(runes[i]) {
			j := i
			for j < len(runes) && isLetterOrDigit(runes[j]) {
				j++
			}
			result = append(result, string(runes[i:j]))
			i = j
		} else if isPunctuation(runes[i]) {
			j := i
			for j < len(runes) && isPunctuation(runes[j]) {
				j++
			}
			result = append(result, string(runes[i:j]))
			i = j
		} else {
			result = append(result, string(runes[i]))
			i++
		}
	}

	return result
}

// sentencePiecePreTokenize replaces spaces with ▁ and splits on ▁ boundaries.
// SentencePiece prepends ▁ to the start of text (sentence boundary).
func sentencePiecePreTokenize(text string) []string {
	const sp = '▁'
	normalized := strings.ReplaceAll(text, " ", string(sp))

	// SentencePiece prepends ▁ if the text doesn't start with it.
	if !strings.HasPrefix(normalized, string(sp)) {
		normalized = string(sp) + normalized
	}

	parts := strings.Split(normalized, string(sp))
	var words []string
	for i, part := range parts {
		if part == "" {
			continue
		}
		if i > 0 {
			words = append(words, string(sp)+part)
		} else {
			words = append(words, part)
		}
	}
	return words
}

func splitToChars(s string) []string {
	var result []string
	for _, r := range s {
		result = append(result, string(r))
	}
	return result
}

func isLetterOrDigit(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

func isPunctuation(r rune) bool {
	return unicode.IsPunct(r) || (r != ' ' && !unicode.IsLetter(r) && !unicode.IsDigit(r) && !unicode.IsSpace(r))
}

// Ensure sorted merge keys for deterministic behavior (unused but kept for future).
var _ = sort.Strings

// --- Special token handling ---

// textSegment represents a piece of text: either a special token or regular text.
type textSegment struct {
	text      string
	isSpecial bool
}

// isSpecialTokenString checks if a vocab token string looks like a special token.
// Special tokens typically match patterns like <|...|>, <s>, </s>, [CLS], [SEP], etc.
func isSpecialTokenString(tok string) bool {
	if len(tok) < 2 {
		return false
	}
	// <|...|> pattern (ChatML, used by Qwen, DeepSeek, MiniCPM)
	if strings.HasPrefix(tok, "<|") && strings.HasSuffix(tok, "|>") {
		return true
	}
	// <s>, </s>, <pad>, <unk>, <bos>, <eos>
	if strings.HasPrefix(tok, "<") && strings.HasSuffix(tok, ">") && len(tok) <= 20 {
		inner := tok[1 : len(tok)-1]
		if inner == "s" || inner == "/s" || strings.HasPrefix(inner, "/") {
			return true
		}
		lower := strings.ToLower(inner)
		switch lower {
		case "pad", "unk", "bos", "eos", "cls", "sep", "mask":
			return true
		}
	}
	// [CLS], [SEP], [PAD], [UNK], [MASK]
	if strings.HasPrefix(tok, "[") && strings.HasSuffix(tok, "]") {
		inner := strings.ToLower(tok[1 : len(tok)-1])
		switch inner {
		case "cls", "sep", "pad", "unk", "mask":
			return true
		}
	}
	return false
}

// splitOnSpecialTokens splits text into segments, identifying special tokens.
// Special tokens are matched greedily (longest match first).
func splitOnSpecialTokens(text string, specialTokens map[string]int32) []textSegment {
	if len(specialTokens) == 0 {
		return []textSegment{{text: text, isSpecial: false}}
	}

	// Sort special token strings by length (longest first) for greedy matching.
	sortedSpecials := make([]string, 0, len(specialTokens))
	for tok := range specialTokens {
		sortedSpecials = append(sortedSpecials, tok)
	}
	sort.Slice(sortedSpecials, func(i, j int) bool {
		return len(sortedSpecials[i]) > len(sortedSpecials[j])
	})

	var segments []textSegment
	remaining := text

	for len(remaining) > 0 {
		found := false
		for _, special := range sortedSpecials {
			if strings.HasPrefix(remaining, special) {
				segments = append(segments, textSegment{text: special, isSpecial: true})
				remaining = remaining[len(special):]
				found = true
				break
			}
		}
		if !found {
			// Find the next special token in the remaining text.
			minIdx := len(remaining)
			for _, special := range sortedSpecials {
				if idx := strings.Index(remaining, special); idx >= 0 && idx < minIdx {
					minIdx = idx
				}
			}
			if minIdx == 0 {
				// Shouldn't happen (we checked prefixes above), but safety.
				minIdx = 1
			}
			if minIdx == len(remaining) {
				// No more special tokens; emit the rest as regular text.
				segments = append(segments, textSegment{text: remaining, isSpecial: false})
				remaining = ""
			} else {
				segments = append(segments, textSegment{text: remaining[:minIdx], isSpecial: false})
				remaining = remaining[minIdx:]
			}
		}
	}

	return segments
}
