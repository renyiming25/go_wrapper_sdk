package common

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"

	"github.com/whybeyoung/go-openai"
)

// truncateForLog returns at most limit runes from s. If truncated, it appends a suffix.
func TruncateForLog(s string, limit int) string {
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}
	return string(runes[:limit]) + fmt.Sprintf("... [truncated, total=%d]", len(runes))
}

func ToString(v any) string {
	if v == nil {
		return "nil"
	}
	res, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("{%T: %v}", v, v)
	}
	return string(res)
}

func jsonStringForLog(v any) string {
	if v == nil {
		return "null"
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("<json.Marshal err: %v type=%T>", err, v)
	}
	return string(b)
}

func JsonStringTruncatedForLog(v any, limit int) string {
	return TruncateForLog(jsonStringForLog(v), limit)
}

// logStreamReq logs ChatCompletionRequest by field; only messages and tools are truncated when too long.
func LogStreamReq(req *openai.ChatCompletionRequest, sid string) {
	if req == nil {
		wLogger.Infow("WrapperWrite streamReq", "sid", sid, "req", nil)
		return
	}
	lim := StreamReqLargeJSONLogRuneLimit
	var seed any
	if req.Seed != nil {
		seed = *req.Seed
	}
	wLogger.Infow("WrapperWrite streamReq",
		"sid", sid,
		"model", req.Model,
		"messages", JsonStringTruncatedForLog(req.Messages, lim),
		"max_tokens", req.MaxTokens,
		"max_completion_tokens", req.MaxCompletionTokens,
		"temperature", req.Temperature,
		"top_p", req.TopP,
		"n", req.N,
		"stream", req.Stream,
		"stop", req.Stop,
		"presence_penalty", req.PresencePenalty,
		"frequency_penalty", req.FrequencyPenalty,
		"response_format", req.ResponseFormat,
		"seed", seed,
		"logit_bias", req.LogitBias,
		"logprobs", req.LogProbs,
		"top_logprobs", req.TopLogProbs,
		"user", req.User,
		"functions", req.Functions,
		"function_call", req.FunctionCall,
		"tools", JsonStringTruncatedForLog(req.Tools, lim),
		"tool_choice", req.ToolChoice,
		"parallel_tool_calls", req.ParallelToolCalls,
		"stream_options", req.StreamOptions,
		"store", req.Store,
		"metadata", req.Metadata,
		"extra_body", req.ExtraBody,
	)
}

func GetFreePort() (int, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()

	addr := listener.Addr().(*net.TCPAddr)
	return addr.Port, nil
}

func WritePortToFile(port int) error {
	file, err := os.Create("/home/aiges/sglangport")
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = fmt.Fprintf(file, "%d", port)
	return err
}

func GetEnvValue(key string) string {
	envStr := os.Getenv(key)
	wLogger.Infof("getEnvValue %s=%s", key, envStr)
	return envStr
}

// isTokenLimitExceededError 检查是否是token超限错误
func IsTokenLimitExceededError(err error) bool {
	if err == nil {
		return false
	}
	errorMsg := err.Error()
	return strings.Contains(errorMsg, "Requested token count exceeds the model's maximum context length") ||
		strings.Contains(errorMsg, "maximum context length") ||
		strings.Contains(errorMsg, "is longer than the model's context length")
}
