package main

import (
	"comwrapper"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"src/inference_wrapper/common"
	"src/utils"
	"strconv"
	"strings"
	"time"
	"unsafe"

	xsfUtils "git.iflytek.com/AIaaS/xsf/utils"
	"github.com/whybeyoung/go-openai"
)

var (
	wLogger                     *xsfUtils.Logger
	requestManager              *common.RequestManager
	cmd                         *exec.Cmd
	promptSearchTemplate        string
	promptSearchTemplateNoIndex string
	pretrainedName              string
	isReasoningModel            bool
	logLevel                           = "info"
	logCount                           = 10
	logSize                            = 30
	logAsync                           = true
	logPath                            = "/log/app/wrapper.log"
	respKey                            = "content"
	httpServerPort                     = 40000
	streamContextTimeoutSeconds        = 5400 * time.Second
	finetuneType                       = "base"
	server                             = "http://127.0.0.1:%d"
	reasoningEffort             string = "high"
	maxStopWords                int    = 16
	customerHeader              string = "x-customer-labels"
	globalEnableThinking        bool   = true
	enableAppIdHeader           bool   = true
	parallelToolCalls           bool   = false
	addWebsearchContent         bool   = false
	enableVllmMetrics           bool   = true
)

// WrapperInit 插件初始化, 全局只调用一次. 本地调试时, cfg参数由aiges.toml提供
func WrapperInit(cfg map[string]string) (err error) {
	fmt.Printf("---- wrapper init ----\n")
	for k, v := range cfg {
		fmt.Printf("config param %s=%s\n", k, v)
	}
	if v, ok := cfg["log_path"]; ok {
		logPath = v
	}
	if v, ok := cfg["log_level"]; ok {
		logLevel = v
	}
	if v, ok := cfg["log_count"]; ok {
		logCount, _ = strconv.Atoi(v)
	}
	if v, ok := cfg["log_size"]; ok {
		logSize, _ = strconv.Atoi(v)
	}
	if cfg["log_async"] == "false" {
		logAsync = false
	}
	if v, ok := cfg["resp_key"]; ok {
		respKey = v
	}
	// 获取搜索模板
	if v, ok := cfg["prompt_search_template"]; ok {
		promptSearchTemplate = v
	}
	// 获取不带索引的搜索模板
	if v, ok := cfg["prompt_search_template_no_index"]; ok {
		promptSearchTemplateNoIndex = v
	}
	if v, ok := cfg["stream_context_timeout_seconds"]; ok {
		if t, err := strconv.ParseInt(v, 10, 64); err == nil {
			streamContextTimeoutSeconds = time.Second * time.Duration(t)
		}
	}
	if v, ok := cfg["max_stop_words"]; ok {
		maxStopWords, _ = strconv.Atoi(v)
	}

	// 创建日志目录
	logDir := filepath.Dir(logPath)
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %v", err)
	}

	wLogger, err = xsfUtils.NewLocalLog(
		xsfUtils.SetAsync(logAsync),
		xsfUtils.SetLevel(logLevel),
		xsfUtils.SetFileName(logPath),
		xsfUtils.SetMaxSize(logSize),
		xsfUtils.SetMaxBackups(logCount),
	)
	if err != nil {
		return fmt.Errorf("loggerErr:%v", err)
	}
	wLogger.Infof("wrapper config:%v", cfg)

	// 设置 common 包的 logger
	common.SetLogger(wLogger)
	common.SetLogLevel(logLevel)

	// 获取搜索模板
	if promptSearchTemplate != "" {
		wLogger.Infow("Using custom prompt search template")
	} else {
		wLogger.Infow("No prompt search template provided")
	}

	// 从环境变量获取配置
	baseModel := common.GetEnvValue("FULL_MODEL_PATH")

	isReasoningModelStr := common.GetEnvValue("IS_REASONING_MODEL")
	if isReasoningModelStr == "true" {
		isReasoningModel = true
	}
	addWebSearchContent := common.GetEnvValue("ADD_WEBSEARCH_CONTENT")
	if addWebSearchContent == "true" {
		addWebsearchContent = true
		common.SetAddWebsearchContent(true)
	}

	reasoningEffortStr := common.GetEnvValue("REASONING_EFFORT")
	if reasoningEffortStr != "" {
		reasoningEffort = reasoningEffortStr
	}

	// 获取端口配置
	port := common.GetEnvValue("HTTP_SERVER_PORT")
	if port != "" {
		httpServerPort, _ = strconv.Atoi(port)
		if httpServerPort == 0 {
			httpServerPort, err = common.GetFreePort()
			if err != nil {
				return fmt.Errorf("getFreePort:%v", err)
			}
		}
	}

	err = common.WritePortToFile(httpServerPort)
	if err != nil {
		return fmt.Errorf("cant write port to file:%v", err)
	}

	// 获取额外参数
	extraArgs := common.GetEnvValue("CMD_EXTRA_ARGS")
	if extraArgs == "" {
		extraArgs = "--gpu-memory-utilization 0.90"
		wLogger.Infow("Using default CMD_ARGS", "args", extraArgs)
	} else {
		wLogger.Infow("Using custom CMD_ARGS", "args", extraArgs)
	}

	// 检查是否启用指标
	// enableMetrics := common.GetEnvValue("ENABLE_METRICS")
	// if enableMetrics == "true" {
	// 	extraArgs += " --enable-metrics"
	// 	wLogger.Infow("Metrics enabled")
	// }

	// 检查多节点模式
	enableMultiNode := common.GetEnvValue("ENABLE_MULTI_NODE_MODE")
	if enableMultiNode == "true" {
		wLogger.Infow("Multi-node mode enabled")

		// 获取组大小 -> vLLM uses --data-parallel-size
		if groupSize := common.GetEnvValue("LWS_GROUP_SIZE"); groupSize != "" {
			extraArgs += fmt.Sprintf(" --data-parallel-size %s", groupSize)
		}

		// 获取leader地址
		if leaderAddr := common.GetEnvValue("LWS_LEADER_ADDRESS"); leaderAddr != "" {
			distPort := common.GetEnvValue("DIST_PORT")
			if distPort == "" {
				distPort = "20000"
			}
			extraArgs += fmt.Sprintf(" --distributed-init-addr %s:%s", leaderAddr, distPort)
		}

		// 获取worker索引
		if workerIndex := common.GetEnvValue("LWS_WORKER_INDEX"); workerIndex != "" {
			extraArgs += fmt.Sprintf(" --node-rank %s", workerIndex)
		}
	}

	pretrainedName = common.GetEnvValue("PRETRAINED_MODEL_NAME")
	finetuneType = common.GetEnvValue("FINETUNE_TYPE")
	globalEnableThinkingStr := common.GetEnvValue("GLOBAL_ENABLE_THINKING")
	if globalEnableThinkingStr == "false" {
		globalEnableThinking = false
	}
	parallelToolCallsStr := common.GetEnvValue("PARALLEL_TOOL_CALLS")
	if parallelToolCallsStr == "true" {
		parallelToolCalls = true
	}
	enableVllmMetricsStr := common.GetEnvValue("ENABLE_VLLM_METRICS")
	if enableVllmMetricsStr == "false" {
		enableVllmMetrics = false
	}

	// 构建完整的启动命令
	args := []string{
		"-m", "vllm.entrypoints.openai.api_server",
		"--model", baseModel,
		"--port", strconv.Itoa(httpServerPort),
		"--trust-remote-code",
		"--host", "0.0.0.0",
	}
	// 添加额外参数
	args = append(args, strings.Fields(extraArgs)...)

	// 启动vllm服务
	cmd = exec.Command("python", args...)
	wLogger.Infow("Starting vllm server", "command", strings.Join(cmd.Args, " "))
	fmt.Printf("Starting vllm server \n")
	// 创建管道捕获输出
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// 启动服务进程 - Start()是非阻塞的
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start vllm server: %v", err)
	}

	// 启动输出监控协程
	go func() {
		_, _ = io.Copy(os.Stdout, stdout)
	}()

	go func() {
		_, _ = io.Copy(os.Stderr, stderr)
	}()

	// 启动监控协程
	go common.MonitorSubprocess(cmd)

	server = fmt.Sprintf(server, httpServerPort)
	// 等待服务就绪
	if err := common.WaitServerReady(fmt.Sprintf("%s/health", server)); err != nil {
		return fmt.Errorf("server failed to start: %v", err)
	}

	// 初始化OpenAI客户端
	serverURL := fmt.Sprintf("%s/v1", server)
	client := common.NewOpenAIClient(serverURL)

	// 初始化请求管理器并保存到全局变量
	requestManager = common.NewRequestManager(client, wLogger)

	if enableVllmMetrics {
		err = utils.UpdatePodMetricsPort(httpServerPort)
		if err != nil {
			wLogger.Errorw("updatePodMetricsPort failed", "error", err)
			return err
		}
	}

	// 获取是否启用 appID 请求级别metadata
	enableAppIdHeaderStr := os.Getenv("ENABLE_APP_ID_HEADER")
	if enableAppIdHeaderStr == "false" {
		enableAppIdHeader = false
		wLogger.Infow("Enable App Id Header is disabled")
	} else {
		wLogger.Infow("Enable App Id Header is enabled")
	}

	// 同步所有配置到 common 包
	common.SetStreamContextTimeout(streamContextTimeoutSeconds)
	common.SetEnableAppIdHeader(enableAppIdHeader)
	common.SetCustomerHeader(customerHeader)
	common.SetGlobalEnableThinking(globalEnableThinking)
	common.SetReasoningEffort(reasoningEffort)
	common.SetMaxStopWords(maxStopWords)
	common.SetParallelToolCalls(parallelToolCalls)
	common.SetPromptSearchTemplate(promptSearchTemplate)
	common.SetPromptSearchTemplateNoIndex(promptSearchTemplateNoIndex)
	common.SetIsReasoningModel(isReasoningModel)

	wLogger.Debugw("WrapperInit successful")
	return
}

// WrapperCreate 插件会话实例创建
func WrapperCreate(usrTag string, params map[string]string, prsIds []int, cb comwrapper.CallBackPtr) (hdl unsafe.Pointer, err error) {
	sid := params["sid"]
	paramStr := ""
	for k, v := range params {
		paramStr += fmt.Sprintf("%s=%s;", k, v)
	}
	wLogger.Debugw("WrapperCreate start", "paramStr", paramStr, "sid", sid)
	appId := params["app_id"]

	// 创建新的实例
	inst := &common.WrapperInst{
		UsrTag:     usrTag,
		Sid:        sid,
		AppId:      appId,
		Client:     requestManager.Client,
		StopQ:      make(chan bool, 2),
		FirstFrame: true,
		Callback:   cb,
		Params:     params,
		Active:     true,
	}

	wLogger.Infow("WrapperCreate successful", "sid", sid, "usrTag", usrTag)
	return unsafe.Pointer(inst), nil
}

func nonStreamingRequest(inst *common.WrapperInst, req *openai.ChatCompletionRequest) error {
	wLogger.Infof("WrapperWrite function_call req:%v, sid:%v\n", common.ToString(req), inst.Sid)
	ctx := context.Background()
	resp, err := inst.Client.OpenaiClient.CreateChatCompletion(ctx, *req)
	wLogger.Infof("WrapperWrite function_call response:%v, sid:%v\n", common.ToString(resp), inst.Sid)
	if err != nil {
		wLogger.Errorw("WrapperWrite function_call CreateChatCompletion error", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}
	var functionCalls []openai.FunctionCall
	if len(resp.Choices) > 0 && len(resp.Choices[0].Message.ToolCalls) > 0 {
		for _, v := range resp.Choices[0].Message.ToolCalls {
			functionCalls = append(functionCalls, openai.FunctionCall{
				Name:      v.Function.Name,
				Arguments: v.Function.Arguments,
			})
		}
	}
	wLogger.Debugf("WrapperWrite function_call toolcalls:%v, sid:%v\n response:%v", common.ToString(functionCalls), inst.Sid, common.ToString(resp))
	// 整理结果
	status := comwrapper.DataEnd
	toolCalls := resp.Choices[0].Message.ToolCalls
	var finishReason string
	if len(resp.Choices) > 0 && resp.Choices[0].FinishReason != "" {
		finishReason = string(resp.Choices[0].FinishReason)
	}
	content, err := common.ResponseContent(status, 0, resp.Choices[0].Message.Content, resp.Choices[0].Message.ReasoningContent, nil, toolCalls, finishReason)
	if err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}

	usage, err := common.ResponseUsage(status, &resp.Usage)
	if err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}
	if err = inst.Callback(inst.UsrTag, []comwrapper.WrapperData{content, usage}, nil); err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback failed", "error", err, "sid", inst.Sid)
		return err
	}
	return nil
}

func openaiFunctionCall(inst *common.WrapperInst, functions []openai.FunctionDefinition, req *openai.ChatCompletionRequest) error {
	openaiTools := make([]openai.Tool, len(functions))
	for i := range functions {
		openaiTools[i] = openai.Tool{
			Type:     "function",
			Function: &functions[i],
		}
	}
	req.Tools = openaiTools
	req.ToolChoice = "auto"
	req.Stream = false
	wLogger.Debugf("WrapperWrite function_call req:%v, sid:%v\n", common.ToString(req), inst.Sid)
	ctx := context.Background()
	resp, err := inst.Client.OpenaiClient.CreateChatCompletion(ctx, *req)
	if err != nil {
		wLogger.Errorw("WrapperWrite function_call CreateChatCompletion error", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}
	var functionCalls []openai.FunctionCall
	if len(resp.Choices) > 0 && len(resp.Choices[0].Message.ToolCalls) > 0 {
		for _, v := range resp.Choices[0].Message.ToolCalls {
			functionCalls = append(functionCalls, openai.FunctionCall{
				Name:      v.Function.Name,
				Arguments: v.Function.Arguments,
			})
		}
	}
	wLogger.Debugf("WrapperWrite function_call toolcalls:%v, sid:%v\n response:%v", common.ToString(functionCalls), inst.Sid, common.ToString(resp))
	// 整理结果
	status := comwrapper.DataEnd
	openaiContent := resp.Choices[0].Message.Content
	if openaiContent == "" {
		openaiContent = " "
	}
	var finishReason string
	if len(resp.Choices) > 0 && resp.Choices[0].FinishReason != "" {
		finishReason = string(resp.Choices[0].FinishReason)
	}
	content, err := common.ResponseContent(status, 0, openaiContent, "", functionCalls, nil, finishReason)
	if err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}

	usage, err := common.ResponseUsage(status, &resp.Usage)
	if err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback", "error", err, "sid", inst.Sid)
		common.ResponseError(inst, err)
		return err
	}
	if err = inst.Callback(inst.UsrTag, []comwrapper.WrapperData{content, usage}, nil); err != nil {
		wLogger.Errorw("WrapperWrite error function_calls callback failed", "error", err, "sid", inst.Sid)
		return err
	}
	return nil
}

// WrapperWrite 数据写入
func WrapperWrite(hdl unsafe.Pointer, req []comwrapper.WrapperData) (err error) {
	inst := (*common.WrapperInst)(hdl)
	if !inst.Active {
		wLogger.Warnw("WrapperWrite called on inactive instance", "sid", inst.Sid)
		return fmt.Errorf("instance is not active")
	}

	if len(req) == 0 {
		wLogger.Debugw("WrapperWrite data is nil", "sid", inst.Sid)
		return nil
	}

	wLogger.Infow("WrapperWrite start", "sid", inst.Sid, "UsrTag", inst.UsrTag, "req", len(req))

	// 检查是否已停止
	select {
	case stopped := <-inst.StopQ:
		if stopped {
			wLogger.Infow("WrapperWrite stopped by signal", "sid", inst.Sid)
			return nil
		}
	default:
	}

	for _, v := range req {
		if v.Key == "__kv_info" {
			continue // 跳过kv_info数据
		}
		// 适配流式请求
		if v.Status != comwrapper.DataEnd {
			if len(inst.StreamContent) == 0 {
				inst.StreamContent = v.Data
			} else {
				inst.StreamContent = append(inst.StreamContent, v.Data...)
			}
			continue
		} else {
			if inst.StreamContent != nil {
				v.Data = append(inst.StreamContent, v.Data...)
			}
		}

		wLogger.Infow("WrapperWrite processing data",
			"data", common.TruncateForLog(string(v.Data), 300),
			"status", v.Status,
			"sid", inst.Sid,
		)

		streamReq, functions, thinking, err := common.BuildStreamReq(inst, v)
		if err != nil {
			wLogger.Errorw("Failed to build stream request", "error", err, "sid", inst.Sid)
			return err
		}

		if !streamReq.Stream {
			return nonStreamingRequest(inst, streamReq)
		}
		// 如果是fuction请求
		if len(functions) > 0 {
			return openaiFunctionCall(inst, functions, streamReq)
		}
		promptTokensLen, resultTokensLen := 0, 0

		common.LogStreamReq(streamReq, inst.Sid)

		// 使用协程处理流式请求
		go func(req *openai.ChatCompletionRequest, status comwrapper.DataStatus) {
			defer func() {
				if r := recover(); r != nil {
					wLogger.Errorw("WrapperWrite panic: ", "sid", inst.Sid, "r", r)
				}
			}()
			wLogger.Infow("WrapperWrite starting stream inference", "sid", inst.Sid)

			ctx, cancel := context.WithTimeout(context.Background(), streamContextTimeoutSeconds)
			inst.Cancel = cancel
			defer cancel()

			stream, err := inst.Client.OpenaiClient.CreateChatCompletionStream(ctx, *req)
			inst.Stream = stream
			if err != nil {
				wLogger.Errorw("WrapperWrite stream error", "error", err, "sid", inst.Sid)
				common.ResponseError(inst, err)
				return
			}
			defer stream.Close()

			index := 0
			fullContent := ""

			status = comwrapper.DataContinue
			// 首帧返回空
			firstFrameContent, err := common.ResponseContent(status, index, "", "", nil, nil, "")
			if err != nil {
				wLogger.Errorw("WrapperWrite error fisrt callback", "error", err, "sid", inst.Sid)
				common.ResponseError(inst, err)
				return
			}
			wLogger.Infof("fisrtFrameContent:%v index:%v,status:%v sid:%v\n", "", index, status, inst.Sid)
			if err := inst.Callback(inst.UsrTag, []comwrapper.WrapperData{firstFrameContent}, nil); err != nil {
				wLogger.Errorw("WrapperWrite error callback failed", "error", err, "sid", inst.Sid)
				return
			}
			index += 1
			var finish_reason string
			for {
				select {
				case <-ctx.Done():
					if ctx.Err() == context.DeadlineExceeded {
						wLogger.Warnw("WrapperWrite stream timeout", "sid", inst.Sid)
						common.ResponseError(inst, fmt.Errorf(common.STREAM_CONNECT_TIMEOUT))
						goto endLoop
					}
					return
				default:
					// 创建响应数据切片
					responseData := make([]comwrapper.WrapperData, 0, 2)

					response, err := stream.Recv()
					if err != nil {
						if err == io.EOF {
							wLogger.Infow("WrapperWrite stream ended normally", "sid", inst.Sid)
							goto endLoop
						}
						// 检查是否是token超限错误
						if common.IsTokenLimitExceededError(err) {
							wLogger.Warnw("Token limit exceeded error detected", "error", err, "sid", inst.Sid)
							err = errors.New(common.ENGINE_TOKEN_LIMIT_EXCEEDED_ERR)
						} else {
							wLogger.Errorw("WrapperWrite read stream error", "error", err, "sid", inst.Sid)
						}
						common.ResponseError(inst, err)
						return
					}
					if index == 1 || logLevel == "debug" {
						wLogger.Infow("WrapperWrite index:1 or logLevel:debug frame content", "response", common.ToString(response), "sid", inst.Sid)
					}
					if len(response.Choices) > 0 {
						var (
							content           string
							reasoning_content string
							tool_calls        []openai.ToolCall
						)
						if response.Choices[0].Delta.ReasoningContent != "" {
							reasoning_content = response.Choices[0].Delta.ReasoningContent
						} else if response.Choices[0].Delta.Content != "" {
							chunk_content := response.Choices[0].Delta.Content
							fullContent += chunk_content
							if index == 1 && strings.HasPrefix(chunk_content, common.R1_THINK_START) {
								thinking = true
								chunks := strings.Split(chunk_content, common.R1_THINK_START)
								reasoning_start_chunk := chunks[1]
								if reasoning_start_chunk != "" {
									reasoning_start_chunk_content, err := common.ResponseContent(status, index, "", reasoning_start_chunk, nil, nil, "")
									wLogger.Debugf("WrapperWrite stream response index:%v, status:%v, reasoning_start_chunk:%v, sid:%v\n", index, status, reasoning_start_chunk, inst.Sid)
									if err != nil {
										wLogger.Errorw("WrapperWrite reasoning_last_chunk error", "error", err, "sid", inst.Sid)
										common.ResponseError(inst, err)
										return
									}
									responseData = append(responseData, reasoning_start_chunk_content)
									if err = inst.Callback(inst.UsrTag, responseData, nil); err != nil {
										wLogger.Errorw("WrapperWrite reasoning_start_chunk_content callback error", "error", err, "sid", inst.Sid)
										return
									}
									index += 1
								}
								continue
							}
							if strings.Contains(chunk_content, common.R1_THINK_END) {
								thinking = false
								chunks := strings.Split(chunk_content, common.R1_THINK_END)
								reasoning_last_chunk := chunks[0]
								answer_start_chunk := chunks[1]

								if reasoning_last_chunk != "" {
									reasoning_last_chunk_content, err := common.ResponseContent(status, index, "", reasoning_last_chunk, nil, nil, "")
									wLogger.Debugf("WrapperWrite stream response index:%v, status:%v, reasoning_last_chunk:%v, sid:%v\n", index, status, reasoning_last_chunk, inst.Sid)
									if err != nil {
										wLogger.Errorw("WrapperWrite reasoning_last_chunk error", "error", err, "sid", inst.Sid)
										common.ResponseError(inst, err)
										return
									}
									responseData = append(responseData, reasoning_last_chunk_content)
									if err = inst.Callback(inst.UsrTag, responseData, nil); err != nil {
										wLogger.Errorw("WrapperWrite reasoning_last_chunk_content callback error", "error", err, "sid", inst.Sid)
										return
									}
									index += 1
								}

								if answer_start_chunk != "" {
									answer_start_chunk_content, err := common.ResponseContent(status, index, answer_start_chunk, "", nil, nil, "")
									if err != nil {
										wLogger.Errorw("WrapperWrite answer_start_chunk error", "error", err, "sid", inst.Sid)
										common.ResponseError(inst, err)
										return
									}
									wLogger.Debugf("WrapperWrite stream response index:%v, status:%v, answer_start_chunk:%v, sid:%v\n", index, status, answer_start_chunk, inst.Sid)
									responseData = []comwrapper.WrapperData{answer_start_chunk_content}
									if err = inst.Callback(inst.UsrTag, responseData, nil); err != nil {
										wLogger.Errorw("WrapperWrite answer_start_chunk_content callback error", "error", err, "sid", inst.Sid)
										return
									}
									index += 1
								}
								continue
							}

							if thinking {
								reasoning_content = chunk_content
							} else {
								content = chunk_content
							}
						} else if len(response.Choices[0].Delta.ToolCalls) > 0 {
							tool_calls = response.Choices[0].Delta.ToolCalls

						} else if response.Choices[0].FinishReason != "" {
							wLogger.Debugf("WrapperWrite stream response finish reason index:%v, status:%v, finish reason:%v, sid:%v\n", index, status, response.Choices[0].FinishReason, inst.Sid)
							finish_reason = string(response.Choices[0].FinishReason)
							continue
						}
						wrapperData, err := common.ResponseContent(status, index, content, reasoning_content, nil, tool_calls, finish_reason)
						if err != nil {
							wLogger.Errorw("WrapperWrite finish reason error", "error", err, "sid", inst.Sid)
							return
						}
						responseData = append(responseData, wrapperData)
						if err = inst.Callback(inst.UsrTag, responseData, nil); err != nil {
							wLogger.Errorw("WrapperWrite finish reason callback error", "error", err, "sid", inst.Sid)
							return
						}
						index += 1

					} else if response.Usage != nil {
						promptTokensLen = response.Usage.PromptTokens
						resultTokensLen = response.Usage.CompletionTokens
						wLogger.Debugf("WrapperWrite stream responseEnd index:%v, promptTokensLen:%v, resultTokensLen:%v sid:%v\n", index, promptTokensLen, resultTokensLen, inst.Sid)
						err = common.ResponseEnd(inst, index, response.Usage, finish_reason)
						if err != nil {
							return
						}
					}

				}
			}

		endLoop:
			wLogger.Infow("WrapperWrite sending end signal", "sid", inst.Sid, "fullContent", fullContent, "in_tokens", promptTokensLen, "out_tokens", resultTokensLen)
			inst.StopQ <- true

		}(streamReq, v.Status)

	}

	return nil
}

// WrapperDestroy 会话资源销毁
func WrapperDestroy(hdl interface{}) (err error) {
	inst := (*common.WrapperInst)(hdl.(unsafe.Pointer))
	wLogger.Infow("WrapperDestroy", "sid", inst.Sid)
	inst.Active = false
	inst.AbortRequest(inst.Sid)
	inst.StopQ <- true

	return nil
}

func WrapperRead(hdl unsafe.Pointer) (respData []comwrapper.WrapperData, err error) {
	return
}

// WrapperFini 插件资源销毁
func WrapperFini() (err error) {
	fmt.Printf("WrapperFini called\n")
	fmt.Printf("WrapperFini cmd:%+v\n", common.ToString(cmd))
	if cmd != nil && cmd.Process != nil {
		if err = cmd.Process.Kill(); err != nil {
			fmt.Printf("WrapperFini err:%v\n", err)
			wLogger.Infof("WrapperFini err:%v", err)
		} else {
			fmt.Printf("WrapperFini cmd kill success\n")
			wLogger.Infof("WrapperFini cmd kill success")
		}
	}
	wLogger.Infof("WarpperFini success")
	return
}

// WrapperVersion 获取版本信息
func WrapperVersion() (version string) {
	return
}

// WrapperDebugInfo 获取调试信息
func WrapperDebugInfo(hdl interface{}) (debug string) {
	return "debug info"
}

// WrapperSetCtrl 设置控制函数
func WrapperSetCtrl(fType comwrapper.CustomFuncType, f interface{}) (err error) {
	switch fType {
	case comwrapper.FuncTraceLog:
		common.TraceFunc = f.(func(usrTag string, key string, value string) (code int))
		fmt.Println("WrapperSetCtrl traceLogFunc set successful.")
	case comwrapper.FuncMeter:
		common.MeterFunc = f.(func(usrTag string, key string, count int) (code int))
		fmt.Println("WrapperSetCtrl meterFunc set successful.")
	case comwrapper.FuncLbExtra:
		common.LbExtraFunc = f.(func(params map[string]string) error)
		fmt.Println("WrapperSetCtrl FuncLbExtra set successful.")
	default:

	}
	return
}

func WrapperExec(usrTag string, params map[string]string, reqData []comwrapper.WrapperData) (respData []comwrapper.WrapperData, err error) {
	return nil, nil
}

// WrapperNotify 插件通知
func WrapperNotify(res comwrapper.WrapperData) (err error) {
	return nil
}
