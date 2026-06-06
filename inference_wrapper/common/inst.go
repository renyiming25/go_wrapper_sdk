package common

import (
	"bytes"
	"comwrapper"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"text/template"
	"time"

	"git.iflytek.com/AIaaS/xsf/utils"
	"github.com/whybeyoung/go-openai"
)

var (
	wLogger             *utils.Logger
	addWebsearchContent bool   = false
	logLevel            string = "info"
)

// SetLogger 设置全局 logger，由 wrapper 调用
func SetLogger(logger *utils.Logger) {
	wLogger = logger
}

// SetLogLevel 设置日志级别
func SetLogLevel(level string) {
	logLevel = level
}

// SetAddWebsearchContent 设置是否添加网络搜索内容
func SetAddWebsearchContent(enable bool) {
	addWebsearchContent = enable
}

// wrapperInst 结构体定义
type WrapperInst struct {
	UsrTag               string
	Sid                  string
	AppId                string
	Client               *OpenAIClient
	StopQ                chan bool
	FirstFrame           bool
	Callback             comwrapper.CallBackPtr
	Params               map[string]string
	Active               bool
	ContinueFinalMessage bool
	StreamContent        []byte
	Stream               *openai.ChatCompletionStream
	Cancel               context.CancelFunc // vLLM: use context cancellation for abort instead of HTTP endpoint
}

// vLLM: abort via context cancellation + stream close (no HTTP abort endpoint)
func (inst *WrapperInst) AbortRequest(sid string) {
	if inst.Cancel != nil {
		inst.Cancel()
	}
	if inst.Stream != nil {
		inst.Stream.Close()
	}
}

// formatMessages 格式化消息，支持搜索模板
func (inst *WrapperInst) FormatMessages(prompt string, promptSearchTemplate string, promptSearchTemplateNoIndex string) ([]Message, []openai.FunctionDefinition) {
	messages, functions := parseMessages(prompt)
	wLogger.Debugf("formatMessages messages: %v\n, functions:%v", messages, functions)

	if promptSearchTemplate == "" && promptSearchTemplateNoIndex == "" {
		return messages, functions
	}

	lastMessage := &messages[len(messages)-1]
	if lastMessage.Role == "assistant" {
		if lastMessage.Prefix != nil && *lastMessage.Prefix {
			inst.ContinueFinalMessage = true
		}
	}

	var lastToolMsg *Message
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "tool" {
			lastToolMsg = &messages[i]
			break
		}
	}
	wLogger.Debugf("formatMessages lastToolMsg: %v\n", lastToolMsg)

	if lastToolMsg == nil {
		return messages, functions
	}

	contentStr := contentToString(lastToolMsg.Content)
	var searchContent []map[string]interface{}
	if err := json.Unmarshal([]byte(contentStr), &searchContent); err != nil {
		wLogger.Errorw("Failed to parse tool message content", "error", err)
		return messages, functions
	}

	if len(searchContent) == 0 {
		return messages, functions
	}

	showRefLabel := false
	if lastToolMsg.ShowRefLabel != nil {
		showRefLabel = *lastToolMsg.ShowRefLabel
	}

	var formattedContent []string
	for _, content := range searchContent {
		formattedText := fmt.Sprintf("[webpage %v begin]\n%v%v\n[webpage %v end]",
			content["index"],
			content["docid"],
			content["document"],
			content["index"])
		formattedContent = append(formattedContent, formattedText)
	}
	wLogger.Debugf("formatMessages formattedContent: %v\n", formattedContent)

	now := time.Now()
	weekdays := []string{"日", "一", "二", "三", "四", "五", "六"}
	currentDate := fmt.Sprintf("%d年%02d月%02d日星期%s",
		now.Year(), now.Month(), now.Day(), weekdays[now.Weekday()])

	addIndex := -1
	if addWebsearchContent {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "tool" {
				for messages[i].Role == "tool" && i > 0 && messages[i-1].Role == "tool" {
					i--
				}
				addIndex = i

				if messages[addIndex-1].Role != "assistant" {
					newMessages := make([]Message, 0, len(messages)+1)
					newMessages = append(newMessages, messages[:addIndex]...)
					newMessages = append(newMessages, Message{
						Role: "assistant",
					})
					newMessages = append(newMessages, messages[addIndex:]...)
					messages = newMessages
					break
				}
			}

		}
	}
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			templateStr := promptSearchTemplate
			if !showRefLabel {
				templateStr = promptSearchTemplateNoIndex
			}

			tmpl, err := template.New("search").Parse(templateStr)
			if err != nil {
				wLogger.Errorw("Failed to parse template", "error", err)
				return messages, functions
			}

			data := struct {
				SearchResults string `json:"search_results"`
				CurDate       string `json:"cur_date"`
				Question      string `json:"question"`
			}{
				SearchResults: strings.Join(formattedContent, "\n"),
				CurDate:       currentDate,
				Question:      contentToString(messages[i].Content),
			}

			var result strings.Builder
			if err := tmpl.Execute(&result, data); err != nil {
				wLogger.Errorw("Failed to execute template", "error", err)
				return messages, functions
			}
			if addWebsearchContent && addIndex != -1 {
				args := map[string]string{
					"content": contentToString(messages[i].Content),
				}
				argsJSON, err := json.Marshal(args)
				if err != nil {
					wLogger.Errorw("Failed to marshal tool call arguments", "error", err)
					return messages, functions
				}
				messages[addIndex].ToolCalls = []openai.ToolCall{
					{
						ID:   "tool_calls",
						Type: "function",
						Function: openai.FunctionCall{
							Name:      "web_search",
							Arguments: string(argsJSON),
						},
					},
				}
			}
			messages[i].Content = result.String()
			break
		}
	}

	return messages, functions
}

// parseMessages 解析消息
func parseMessages(prompt string) ([]Message, []openai.FunctionDefinition) {
	var messages []Message
	if err := json.Unmarshal([]byte(prompt), &messages); err == nil {
		for _, msg := range messages {
			if msg.Role == "" || msg.Content == nil {
				wLogger.Errorw("parseMessages Invalid message format", "message", msg)
				return []Message{{
					Role:    "user",
					Content: prompt,
				}}, nil
			}
		}
		return messages, nil
	}
	wLogger.Debugw("parseMessages try sparkMsg")
	var sparkMsg struct {
		Messages  []Message                   `json:"messages"`
		Functions []openai.FunctionDefinition `json:"functions"`
	}
	if err := json.Unmarshal([]byte(prompt), &sparkMsg); err == nil {
		return sparkMsg.Messages, sparkMsg.Functions
	}

	wLogger.Debugw("parseMessages Using plain text as message", "prompt", prompt)
	return []Message{{
		Role:    "user",
		Content: prompt,
	}}, nil
}

// contentToString 安全地将 Content 转换为字符串
func contentToString(content interface{}) string {
	if content == nil {
		return ""
	}

	if str, ok := content.(string); ok {
		return str
	}

	if arr, ok := content.([]interface{}); ok {
		var parts []string
		for _, item := range arr {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if text, ok := itemMap["text"].(string); ok {
					parts = append(parts, text)
				}
			} else if str, ok := item.(string); ok {
				parts = append(parts, str)
			}
		}
		return strings.Join(parts, "")
	}

	return fmt.Sprintf("%v", content)
}

// WarmupRequest 发送预热请求
func WarmupRequest(url string, warmupDataPath string) error {
	wLogger.Infow("Starting warmup process", "url", url, "warmupDataPath", warmupDataPath)

	data, err := os.ReadFile(warmupDataPath)
	if err != nil {
		wLogger.Errorw("Failed to read warmup data file", "error", err, "path", warmupDataPath)
		return fmt.Errorf("failed to read warmup data file: %v", err)
	}

	var warmupPrompts []WarmupPrompt
	if err := json.Unmarshal(data, &warmupPrompts); err != nil {
		wLogger.Errorw("Failed to parse warmup data", "error", err)
		return fmt.Errorf("failed to parse warmup data: %v", err)
	}

	if len(warmupPrompts) == 0 {
		wLogger.Warnw("No warmup prompts found")
		return nil
	}

	wLogger.Infow("Loaded warmup prompts", "count", len(warmupPrompts))

	successCount := 0
	totalCount := len(warmupPrompts)
	wg := sync.WaitGroup{}
	var successMutex sync.Mutex

	for i, prompt := range warmupPrompts {
		wg.Add(1)
		go func(i int, prompt WarmupPrompt) {
			defer wg.Done()
			payload := map[string]interface{}{
				"model": DEFAULT_MODEL_NAME,
				"messages": []map[string]interface{}{
					{
						"role":    "user",
						"content": prompt.Prompt,
					},
				},
				"max_tokens":  prompt.MaxTokens,
				"temperature": prompt.Temperature,
				"stream":      false,
			}

			startTime := time.Now()
			resp, err := PostRequest(url, payload)
			duration := time.Since(startTime)

			if err != nil {
				wLogger.Errorw("Warmup request failed", "index", i+1, "error", err, "prompt", prompt.Prompt)
			} else {
				successMutex.Lock()
				successCount++
				successMutex.Unlock()
				wLogger.Infow("Warmup request successful", "index", i+1, "duration", duration, "response_length", len(resp))
			}
		}(i, prompt)
	}
	wg.Wait()

	successRate := float64(successCount) / float64(totalCount)
	wLogger.Infow("Warmup completed", "success_count", successCount, "total_count", totalCount, "success_rate", successRate)

	if successRate < 0.8 {
		wLogger.Errorw("Warmup success rate too low", "success_rate", successRate)
		return fmt.Errorf("warmup success rate too low: %.2f", successRate)
	}

	return nil
}

// PostRequest 发送POST请求（使用默认参数）
func PostRequest(url string, jsonData map[string]interface{}) ([]byte, error) {
	return PostRequestWithOptions(url, jsonData, nil, 3, 60*time.Second)
}

// PostRequestWithOptions 发送POST请求（自定义参数）
func PostRequestWithOptions(url string, jsonData map[string]interface{}, headers map[string]string, maxRetries int, timeout time.Duration) ([]byte, error) {
	client := &http.Client{
		Timeout: timeout,
	}

	var requestBody []byte
	var err error

	if jsonData != nil {
		requestBody, err = json.Marshal(jsonData)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal json data: %v", err)
		}
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			wLogger.Debugw("Retrying POST request", "attempt", attempt, "url", url)
			time.Sleep(1 * time.Second)
		}

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
		if err != nil {
			lastErr = fmt.Errorf("failed to create request: %v", err)
			continue
		}

		if jsonData != nil {
			req.Header.Set("Content-Type", "application/json")
		}

		for key, value := range headers {
			req.Header.Set(key, value)
		}

		resp, err := client.Do(req)
		if logLevel == "debug" {
			errorMsg := ""
			if err != nil {
				errorMsg = err.Error()
			}
			wLogger.Debugw("Post process", "req", ToString(req), "resp", ToString(resp), "error", errorMsg)
		}

		if err != nil {
			lastErr = fmt.Errorf("request failed (attempt %d): %v", attempt+1, err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("failed to read response body (attempt %d): %v", attempt+1, err)
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			wLogger.Debugw("POST request successful", "url", url, "status", resp.StatusCode, "attempt", attempt+1)
			return body, nil
		}

		lastErr = fmt.Errorf("HTTP error (attempt %d): %s - %s", attempt+1, resp.Status, string(body))

		if resp.StatusCode >= 400 && resp.StatusCode < 500 {
			wLogger.Warnw("Client error, not retrying", "status", resp.StatusCode, "body", string(body))
			break
		}
	}

	return nil, fmt.Errorf("all retry attempts failed. Last error: %v", lastErr)
}

// waitServerReady 等待服务器就绪
func WaitServerReady(serverURL string) error {
	httpClient := http.Client{
		Timeout: HTTP_SERVER_REQUEST_TIMEOUT,
	}
	retries := 0
	timeBegin := time.Now()
	for {
		resp, err := httpClient.Get(serverURL)
		wLogger.Debugw("Server resp", "resp", resp, "err", err)
		if err == nil && resp.StatusCode == http.StatusOK {
			wLogger.Debugw("Server ready", "url", serverURL)
			return nil
		}

		if resp != nil {
			resp.Body.Close()
		}

		retries++
		timeInerval := time.Since(timeBegin)
		if timeInerval >= HTTP_SERVER_MAX_RETRY_TIME {
			break
		}

		wLogger.Debugw("Server not ready, retrying...", "url", serverURL, "attempt", retries)
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("server failed to start after %d attempts", retries)
}

// monitorSubprocess 监控子进程
func MonitorSubprocess(cmd *exec.Cmd) {
	wLogger.Debugw("Starting Vllm process monitor...")

	err := cmd.Wait()
	if err != nil {
		fmt.Printf("Vllm process exited with error:%v\n", err)
		wLogger.Errorw("Vllm process exited with error", "error", err)
	} else {
		fmt.Printf("Vllm process exited normally\n")
		wLogger.Debugw("Vllm process exited normally")
	}
}
