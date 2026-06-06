package common

import (
	"time"
)

const (
	R1_THINK_START                  = "<think>"
	R1_THINK_END                    = "</think>"
	GPT_THINK_START                 = "analysis"
	GPT_THINK_END_ASSISTANT         = "assistant"
	GPT_THINK_END_FINAL             = "final"
	HTTP_SERVER_REQUEST_TIMEOUT     = time.Second * 10
	HTTP_SERVER_MAX_RETRY_TIME      = time.Minute * 60
	STREAM_CONNECT_TIMEOUT          = "stream connect timeout"
	ENGINE_TOKEN_LIMIT_EXCEEDED_ERR = "ENGINE_TOKEN_LIMIT_EXCEEDED;code=2001"
	ENGINE_REQUEST_ERR              = "ENGINE_REQUEST_ERR;code=2002"
	DEFAULT_MODEL_NAME              = "default"
)

// streamReqLargeJSONLogRuneLimit truncates only the JSON-serialized messages/tools blobs in debug logs.
const StreamReqLargeJSONLogRuneLimit = 4096

var TraceFunc func(usrTag string, key string, value string) (code int)
var MeterFunc func(usrTag string, key string, count int) (code int)
var LbExtraFunc func(params map[string]string) error
