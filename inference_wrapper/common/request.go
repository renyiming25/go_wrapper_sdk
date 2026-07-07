package common

import (
	"comwrapper"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"git.iflytek.com/AIaaS/xsf/utils"
	"github.com/whybeyoung/go-openai"
)

var (
	streamContextTimeoutSeconds        = 5400 * time.Second
	enableAppIdHeader           bool   = true
	customerHeader              string = "x-customer-labels"
	globalEnableThinking               = true
	reasoningEffort                    = "high"
	maxStopWords                int    = 16
	parallelToolCalls           bool   = false
	promptSearchTemplate        string
	promptSearchTemplateNoIndex string
	isReasoningModel            bool
	servedModelName             string = DEFAULT_MODEL_NAME
)

// Setter functions to sync configuration from wrapper
func SetStreamContextTimeout(timeout time.Duration) {
	streamContextTimeoutSeconds = timeout
}

func SetEnableAppIdHeader(enable bool) {
	enableAppIdHeader = enable
}

func SetCustomerHeader(header string) {
	customerHeader = header
}

func SetGlobalEnableThinking(enable bool) {
	globalEnableThinking = enable
}

func SetReasoningEffort(effort string) {
	reasoningEffort = effort
}

func SetMaxStopWords(max int) {
	maxStopWords = max
}

func SetParallelToolCalls(enable bool) {
	parallelToolCalls = enable
}

func SetPromptSearchTemplate(template string) {
	promptSearchTemplate = template
}

func SetPromptSearchTemplateNoIndex(template string) {
	promptSearchTemplateNoIndex = template
}

func SetIsReasoningModel(enable bool) {
	isReasoningModel = enable
}

func SetServedModelName(modelName string) {
	servedModelName = modelName
}

// NewRequestManager 创建请求管理器
func NewRequestManager(client *OpenAIClient, logger *utils.Logger) *RequestManager {
	return &RequestManager{
		Client: client,
		Logger: logger,
	}
}

func BuildStreamReq(inst *WrapperInst, req comwrapper.WrapperData) (*openai.ChatCompletionRequest, []openai.FunctionDefinition, bool, error) {
	// 从params中获取参数, 仅用户显式传入时才设置
	var (
		temperature *float32
		maxTokens   int
		topP        *float32
		topK        int
		stop        []string
	)

	if tempStr, ok := inst.Params["temperature"]; ok {
		if t, err := strconv.ParseFloat(tempStr, 64); err == nil {
			v := float32(t)
			temperature = &v
		} else {
			wLogger.Warnw("Invalid temperature value", "value", tempStr, "sid", inst.Sid)
		}
	}

	if tokensStr, ok := inst.Params["max_tokens"]; ok {
		if t, err := strconv.Atoi(tokensStr); err == nil {
			maxTokens = t
		} else {
			wLogger.Warnw("Invalid max_tokens value", "value", tokensStr, "sid", inst.Sid)
		}
	}

	if tpStr, ok := inst.Params["top_p"]; ok {
		if t, err := strconv.ParseFloat(tpStr, 32); err == nil {
			v := float32(t)
			topP = &v
		} else {
			wLogger.Warnw("Invalid top_p value", "value", tpStr, "sid", inst.Sid)
		}
	}

	if tkStr, ok := inst.Params["top_k"]; ok {
		if t, err := strconv.Atoi(tkStr); err == nil {
			topK = t
		} else {
			wLogger.Warnw("Invalid top_k value", "value", tkStr, "sid", inst.Sid)
		}
	}

	streamReq := &openai.ChatCompletionRequest{
		Model: servedModelName,
	}

	wLogger.Infow("WrapperWrite request parameters",
		"sid", inst.Sid,
		"param", inst.Params,
	)
	if maxTokens > 0 {
		streamReq.MaxTokens = maxTokens
	}
	if temperature != nil {
		streamReq.Temperature = temperature
	}
	if topP != nil {
		streamReq.TopP = topP
	}
	streamReq.Stream = true
	streamReq.StreamOptions = &openai.StreamOptions{
		IncludeUsage: true,
	}
	// 设定 appID 请求级别，用于统计
	if inst.AppId != "" && enableAppIdHeader {
		appIdHeader := CustomerLabel{
			AppId: inst.AppId,
		}
		appIdHeaderStr, err := json.Marshal(appIdHeader)
		if err != nil {
			wLogger.Errorw("WrapperWrite marshal appIdHeader error", "error", err, "sid", inst.Sid, "appId", inst.AppId)
		}
		streamReq.Metadata = map[string]string{
			customerHeader: string(appIdHeaderStr),
		}
	}

	enableThinking := globalEnableThinking
	if enableThinkingParam, ok := inst.Params["enable_thinking"]; ok {
		if enableThinkingParam == "false" {
			enableThinking = false
		} else {
			enableThinking = true
		}
	}
	streamReq.ExtraBody = map[string]any{
		"chat_template_kwargs": map[string]interface{}{
			"enable_thinking": enableThinking,
			"thinking":        enableThinking,
		},
		"rid": inst.Sid,
	}

	// 从params中获取 extra_body
	extraBodyStr := ""
	if v, ok := inst.Params["extra_body"]; ok {
		extraBodyStr = v
	}
	// 解析 extraBodyStr
	var extraParams ExtraParams
	if extraBodyStr != "" {
		if err := json.Unmarshal([]byte(extraBodyStr), &extraParams); err != nil {
			wLogger.Errorw("WrapperWrite unmarshal extra_parms error", "error", err, "sid", inst.Sid, "extra_parms", extraBodyStr)
		}
	}
	if extraParams.ReasoningEffort != "" {
		streamReq.ExtraBody["reasoning_effort"] = extraParams.ReasoningEffort
	} else {
		streamReq.ExtraBody["reasoning_effort"] = reasoningEffort
	}
	// 解析 extra_parms 的 response_format string 反序列化 ChatCompletionResponseFormat格式
	var responseFormat openai.ChatCompletionResponseFormat
	if extraParams.ResponseFormat != nil {
		responseFormat.Type = openai.ChatCompletionResponseFormatType(extraParams.ResponseFormat.Type)
		if extraParams.ResponseFormat.JSONSchema != nil {
			// 将 schema 转换为 json.RawMessage
			schemaBytes, err := json.Marshal(extraParams.ResponseFormat.JSONSchema.Schema)
			if err != nil {
				wLogger.Errorw("WrapperWrite marshal schema error", "error", err, "sid", inst.Sid)
			} else {
				// 创建一个自定义的 Marshaler 类型
				responseFormat.JSONSchema = &openai.ChatCompletionResponseFormatJSONSchema{
					Name:        extraParams.ResponseFormat.JSONSchema.Name,
					Description: extraParams.ResponseFormat.JSONSchema.Description,
					Schema:      SchemaMarshaler{Data: schemaBytes},
					Strict:      extraParams.ResponseFormat.JSONSchema.Strict,
				}
			}
		}
	}
	if responseFormat.Type != "" {
		streamReq.ResponseFormat = &responseFormat
	}
	if extraParams.FrequencyPenalty != nil {
		streamReq.FrequencyPenalty = *extraParams.FrequencyPenalty
	}
	if extraParams.PresencePenalty != nil {
		streamReq.PresencePenalty = *extraParams.PresencePenalty
	}
	if extraParams.ContinueFinalMessage {
		inst.ContinueFinalMessage = true
	}
	if extraParams.SkipSpecialTokens != nil {
		streamReq.ExtraBody["skip_special_tokens"] = *extraParams.SkipSpecialTokens
	}
	if topK > 0 {
		streamReq.ExtraBody["top_k"] = topK
	}
	if extraParams.RepetitionPenalty != nil {
		streamReq.ExtraBody["repetition_penalty"] = *extraParams.RepetitionPenalty
	}
	if len(extraParams.Stop) > 0 {
		if len(extraParams.Stop) > maxStopWords {
			wLogger.Warnw("WrapperWrite stop words over limit", "stop", extraParams.Stop, "maxStopWords", maxStopWords, "sid", inst.Sid)
			stop = extraParams.Stop[:maxStopWords]
		} else {
			stop = extraParams.Stop
			wLogger.Warnw("WrapperWrite stop words", "stop", extraParams.Stop, "sid", inst.Sid)
		}
	}

	if toolsStr, ok := inst.Params["tools"]; ok {
		tools := make([]openai.Tool, 0)
		if err := json.Unmarshal([]byte(toolsStr), &tools); err != nil {
			wLogger.Errorw("WrapperWrite unmarshal tools error", "error", err, "sid", inst.Sid, "tools", toolsStr)
		}
		streamReq.Tools = tools
		if toolChoiceStr, ok := inst.Params["tool_choice"]; ok && toolChoiceStr != "" {
			wLogger.Infow("WrapperWrite toolChoiceStr", "sid", inst.Sid, "toolChoiceStr", toolChoiceStr)
			if strings.HasPrefix(toolChoiceStr, "{") {
				toolChoice := openai.ToolChoice{}
				err := json.Unmarshal([]byte(toolChoiceStr), &toolChoice)
				if err != nil {
					wLogger.Errorw("WrapperWrite unmarshal toolChoiceStr error", "error", err, "sid", inst.Sid, "toolChoiceStr", toolChoiceStr)
				}
				streamReq.ToolChoice = toolChoice
			} else {
				toolChoiceStr = strings.ReplaceAll(toolChoiceStr, "\"", "")
				streamReq.ToolChoice = toolChoiceStr
			}
		} else {
			streamReq.ToolChoice = "auto"
		}
		if parallelToolCallsStr, ok := inst.Params["parallel_tool_calls"]; ok {
			if parallelToolCallsStr == "true" {
				streamReq.ParallelToolCalls = true
			} else {
				streamReq.ParallelToolCalls = false
			}
		} else {
			streamReq.ParallelToolCalls = parallelToolCalls
		}
	}
	if extraParams.LogitBias != nil {
		streamReq.LogitBias = extraParams.LogitBias
	}

	openaiMsgs, functions := inst.FormatMessages(string(req.Data), promptSearchTemplate, promptSearchTemplateNoIndex)
	thinking := false
	if isReasoningModel {
		lastMsg := openaiMsgs[len(openaiMsgs)-1]
		if lastMsg.Role != "assistant" {
			openaiMsgs = append(openaiMsgs, Message{
				Role:    "assistant",
				Content: R1_THINK_START + "\n",
			})
			thinking = true
		}
	}
	if inst.ContinueFinalMessage {
		streamReq.ExtraBody["continue_final_message"] = true
	}
	if len(stop) > 0 {
		streamReq.Stop = stop
	}
	messages, err := convertToOpenAIMessages(openaiMsgs)
	if err != nil {
		wLogger.Errorw("Failed to convert messages", "error", err, "sid", inst.Sid)
	}
	streamReq.Messages = messages
	return streamReq, functions, thinking, err
}

func convertToOpenAIMessages(messages []Message) ([]openai.ChatCompletionMessage, error) {
	openAIMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openAIMessages[i] = openai.ChatCompletionMessage{
			Role:       msg.Role,
			ToolCalls:  msg.ToolCalls,
			ToolCallID: msg.ToolCallID,
		}
		content := ""
		if msg.Content != nil {
			switch v := msg.Content.(type) {
			case string:
				content = v
				openAIMessages[i].Content = content

			case []interface{}:
				multiContent := make([]openai.ChatMessagePart, 0)
				wLogger.Infow("convertToOpenAIMessages multi content", "content", JsonStringTruncatedForLog(msg.Content, StreamReqLargeJSONLogRuneLimit), "contentType", fmt.Sprintf("%T", msg.Content))

				for _, item := range v {
					jsonBytes, marshalErr := json.Marshal(item)
					if marshalErr != nil {
						wLogger.Errorw("Failed to marshal content item", "error", marshalErr, "item_type", fmt.Sprintf("%T", item))
						return nil, fmt.Errorf("failed to marshal content item: %v", marshalErr)
					}
					var part openai.ChatMessagePart
					if unmarshalErr := json.Unmarshal(jsonBytes, &part); unmarshalErr != nil {
						wLogger.Errorw("Failed to unmarshal content item", "error", unmarshalErr, "item", string(jsonBytes))
						return nil, fmt.Errorf("failed to unmarshal content item: %v", unmarshalErr)
					}
					multiContent = append(multiContent, part)
				}
				openAIMessages[i].MultiContent = multiContent

			default:
				if jsonBytes, err := json.Marshal(v); err == nil {
					content = string(jsonBytes)
					openAIMessages[i].Content = content
				}
			}
		}
	}
	return openAIMessages, nil
}

// NewOpenAIClient 创建OpenAI客户端
func NewOpenAIClient(baseURL string) *OpenAIClient {
	config := openai.DefaultConfig("maas")
	config.BaseURL = baseURL
	config.HTTPClient = &http.Client{
		Timeout: streamContextTimeoutSeconds,
		Transport: &http.Transport{
			MaxIdleConns:        6000,
			MaxIdleConnsPerHost: 3000,
			MaxConnsPerHost:     3000,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	return &OpenAIClient{
		OpenaiClient: openai.NewClientWithConfig(config),
	}
}

func parseFunctions(funcStr string) []*openai.FunctionDefinition {
	var functions []*openai.FunctionDefinition
	if err := json.Unmarshal([]byte(funcStr), &functions); err != nil {
		wLogger.Errorw("Invalid functions format", "functions", funcStr)
	}
	return functions
}
