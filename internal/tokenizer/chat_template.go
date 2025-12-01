package tokenizer

import (
	"fmt"
	"strings"
)

// ChatMLTemplate implements the ChatML format used by OpenAI and DeepSeek.
//
// Format: <|im_start|>role\ncontent<|im_end|>.
type ChatMLTemplate struct{}

// NewChatMLTemplate creates a new ChatML template.
func NewChatMLTemplate() *ChatMLTemplate {
	return &ChatMLTemplate{}
}

// Apply formats messages in ChatML format.
func (t *ChatMLTemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder

	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|im_end|>\n")
	}

	// Add assistant start token for generation.
	sb.WriteString("<|im_start|>assistant\n")

	return sb.String()
}

// Name returns the template name.
func (t *ChatMLTemplate) Name() string {
	return "ChatML"
}

// LLaMATemplate implements the LLaMA chat format.
//
// Format: [INST] user message [/INST] assistant response.
type LLaMATemplate struct {
	bosToken string
	eosToken string
}

// NewLLaMATemplate creates a new LLaMA chat template.
func NewLLaMATemplate() *LLaMATemplate {
	return &LLaMATemplate{
		bosToken: "<s>",
		eosToken: "</s>",
	}
}

// Apply formats messages in LLaMA format.
func (t *LLaMATemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder

	sb.WriteString(t.bosToken)

	// Separate system prompt from conversation.
	var systemPrompt string
	var conversation []ChatMessage

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
		} else {
			conversation = append(conversation, msg)
		}
	}

	// Format conversation turns.
	for i := 0; i < len(conversation); i++ {
		msg := conversation[i]

		switch msg.Role {
		case "user":
			sb.WriteString("[INST] ")
			if i == 0 && systemPrompt != "" {
				// Include system prompt in first user message.
				sb.WriteString("<<SYS>>\n")
				sb.WriteString(systemPrompt)
				sb.WriteString("\n<</SYS>>\n\n")
			}
			sb.WriteString(msg.Content)
			sb.WriteString(" [/INST]")
		case "assistant":
			sb.WriteString(" ")
			sb.WriteString(msg.Content)
			sb.WriteString(t.eosToken)
			sb.WriteString(t.bosToken)
		}
	}

	return sb.String()
}

// Name returns the template name.
func (t *LLaMATemplate) Name() string {
	return "LLaMA"
}

// MistralTemplate implements the Mistral chat format.
//
// Similar to LLaMA but with slight variations.
type MistralTemplate struct {
	bosToken string
	eosToken string
}

// NewMistralTemplate creates a new Mistral chat template.
func NewMistralTemplate() *MistralTemplate {
	return &MistralTemplate{
		bosToken: "<s>",
		eosToken: "</s>",
	}
}

// Apply formats messages in Mistral format.
func (t *MistralTemplate) Apply(messages []ChatMessage) string {
	var sb strings.Builder

	sb.WriteString(t.bosToken)

	for i, msg := range messages {
		switch msg.Role {
		case "user":
			sb.WriteString("[INST] ")
			sb.WriteString(msg.Content)
			sb.WriteString(" [/INST]")
		case "assistant":
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(msg.Content)
			sb.WriteString(t.eosToken)
			if i < len(messages)-1 {
				sb.WriteString(t.bosToken)
			}
		case "system":
			// Mistral doesn't have explicit system role, prepend to first user message.
			// We'll handle this in a simplified way.
			if i == 0 {
				sb.WriteString("[INST] ")
				sb.WriteString(msg.Content)
				sb.WriteString(" [/INST]")
			}
		}
	}

	return sb.String()
}

// Name returns the template name.
func (t *MistralTemplate) Name() string {
	return "Mistral"
}

// GetChatTemplate returns a chat template by name.
func GetChatTemplate(name string) (ChatTemplate, error) {
	switch strings.ToLower(name) {
	case "chatml":
		return NewChatMLTemplate(), nil
	case "llama":
		return NewLLaMATemplate(), nil
	case "mistral":
		return NewMistralTemplate(), nil
	default:
		return nil, fmt.Errorf("unknown chat template: %s", name)
	}
}
