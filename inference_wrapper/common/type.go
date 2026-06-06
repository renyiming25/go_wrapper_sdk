package common

import (
	"sync"

	"git.iflytek.com/AIaaS/xsf/utils"
	"github.com/whybeyoung/go-openai"
)

// OpenAIClient OpenAI客户端
type OpenAIClient struct {
	OpenaiClient *openai.Client
}

// RequestManager 请求管理器
type RequestManager struct {
	Client      *OpenAIClient
	Logger      *utils.Logger
	RequestLock sync.RWMutex
}

// 额外参数
type ExtraParams struct {
	ResponseFormat *struct {
		Type       string `json:"type,omitempty"`
		JSONSchema *struct {
			Name        string                 `json:"name"`
			Description string                 `json:"description,omitempty"`
			Schema      map[string]interface{} `json:"schema"`
			Strict      bool                   `json:"strict"`
		} `json:"json_schema,omitempty"`
	} `json:"response_format,omitempty"`
	LogitBias            map[string]int `json:"logit_bias,omitempty"`
	ReasoningEffort      string         `json:"reasoning_effort,omitempty"`
	FrequencyPenalty     *float32       `json:"frequency_penalty,omitempty"`
	PresencePenalty      *float32       `json:"presence_penalty,omitempty"`
	ContinueFinalMessage bool           `json:"continue_final_message,omitempty"`
	Stop                 []string       `json:"stop,omitempty"`
	SkipSpecialTokens    *bool          `json:"skip_special_tokens,omitempty"`
}

// app_Id
type CustomerLabel struct {
	AppId string `json:"app_id,omitempty"`
}

// Message 消息结构
type Message struct {
	Role         string            `json:"role"`
	Content      interface{}       `json:"content"`
	ShowRefLabel *bool             `json:"show_ref_label,omitempty"`
	Prefix       *bool             `json:"prefix,omitempty"`
	ToolCalls    []openai.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID   string            `json:"tool_call_id,omitempty"`
}

// Tool 工具结构
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function 函数结构
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ChatCompletionRequest 聊天完成请求
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
}

// Choice 选择结构
type Choice struct {
	Index   int     `json:"index"`
	Message Message `json:"message"`
}

// Usage 使用量结构
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse 聊天完成响应
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// schemaMarshaler 自定义的 Marshaler 类型
type SchemaMarshaler struct {
	Data []byte
}

func (s SchemaMarshaler) MarshalJSON() ([]byte, error) {
	return s.Data, nil
}

// WarmupPrompt 预热请求数据结构
type WarmupPrompt struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Type        string  `json:"type"`
}
