package common

import (
	"encoding/json"
	"comwrapper"

	"github.com/whybeyoung/go-openai"
)

func ResponseContent(status comwrapper.DataStatus, index int, text string, resoning_content string, function_call []openai.FunctionCall, tool_calls []openai.ToolCall, finish_reason string) (comwrapper.WrapperData, error) {
	wLogger.Debugf("WrapperWrite responseContent status:%v, index:%v, text:%v, resoning_content:%v, function_call:%v, tool_calls:%v, finish_reason:%v\n", status, index, text, resoning_content, function_call, tool_calls, finish_reason)
	choice := map[string]interface{}{
		"content":           text,
		"reasoning_content": resoning_content,
		"index":             0,
	}
	if index == 0 {
		choice["role"] = "assistant"
	}
	if len(function_call) > 0 {
		choice["function_call"] = function_call
	}
	if len(tool_calls) > 0 {
		choice["tool_calls"] = tool_calls
	}
	if finish_reason != "" {
		choice["finish_reason"] = finish_reason
	}
	result := map[string]interface{}{
		"choices": []map[string]interface{}{
			choice,
		},
		"question_type": "",
	}
	data, err := json.Marshal(result)

	return comwrapper.WrapperData{
		Key:      "content",
		Data:     data,
		Desc:     nil,
		Encoding: "utf-8",
		Type:     comwrapper.DataText,
		Status:   status,
	}, err
}

func ResponseUsage(status comwrapper.DataStatus, usage *openai.Usage) (comwrapper.WrapperData, error) {
	result := map[string]interface{}{
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
		"question_tokens":   4,
	}
	if usage.PromptTokensDetails != nil {
		result["prompt_tokens_details"] = usage.PromptTokensDetails
	}
	if usage.CompletionTokensDetails != nil {
		result["completion_tokens_details"] = usage.CompletionTokensDetails
	}
	data, err := json.Marshal(result)

	return comwrapper.WrapperData{
		Key:      "usage",
		Data:     data,
		Desc:     nil,
		Encoding: "utf-8",
		Type:     comwrapper.DataText,
		Status:   status,
	}, err
}

func ResponseError(inst *WrapperInst, err error) {
	wLogger.Errorf("WrapperWrite stream err:%v, sid:%v\n", err.Error(), inst.Sid)

	if err = inst.Callback(inst.UsrTag, nil, err); err != nil {
		wLogger.Errorw("WrapperWrite error callback failed", "error", err, "sid", inst.Sid)
	}
}

func ResponseEnd(inst *WrapperInst, index int, usage *openai.Usage, finish_reason string) error {
	status := comwrapper.DataEnd
	usageWrapperData, err := ResponseUsage(status, usage)
	if err != nil {
		ResponseError(inst, err)
		return err
	}
	content, err := ResponseContent(status, index, "", "", nil, nil, finish_reason)
	if err != nil {
		ResponseError(inst, err)
		return err
	}
	responseData := []comwrapper.WrapperData{content, usageWrapperData}
	wLogger.Infof("WrapperWrite stream responseEnd index:%v, prompt_tokens_len:%v, result_tokens_len:%v,responseData:%v, sid:%v\n", index, usage.PromptTokens, usage.CompletionTokens, responseData, inst.Sid)
	if err = inst.Callback(inst.UsrTag, responseData, nil); err != nil {
		wLogger.Errorw("WrapperWrite end callback error", "error", err, "sid", inst.Sid)
		return err
	}
	return nil
}