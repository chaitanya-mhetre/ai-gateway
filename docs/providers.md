# Provider wire formats (what the adapters translate)

Checked against the providers' docs in Sep 2026. Formats change, so re-verify before relying on details.

| | OpenAI Chat Completions | Anthropic Messages | Gemini generateContent | Ollama /api/chat |
|---|---|---|---|---|
| Auth | `Authorization: Bearer` | `x-api-key` + `anthropic-version` | `x-goog-api-key` | none (local) |
| System prompt | `system` message | top-level `system` | `systemInstruction.parts` | `system` message |
| Roles | system/user/assistant/tool | user/assistant (must alternate) | user/model | system/user/assistant/tool |
| Max tokens | `max_completion_tokens` | `max_tokens` (**required**) | `generationConfig.maxOutputTokens` | `options.num_predict` |
| Tool call out | `tool_calls[{id, function{name, arguments:str}}]` | content block `tool_use{id,name,input:obj}` | part `functionCall{name,args:obj}` (id optional) | `tool_calls[{function{name,arguments:obj}}]` (no id) |
| Tool result in | `role: tool`, `tool_call_id` | `tool_result` block inside a **user** message | `functionResponse{name,response}` part (by **name**) | `role: tool` |
| Stop reasons | stop/length/tool_calls/content_filter | end_turn/max_tokens/tool_use/stop_sequence/refusal | STOP/MAX_TOKENS/SAFETY/... | done_reason stop/length |
| Usage | prompt/completion, `prompt_tokens_details.cached_tokens` | input (excl. cache) + cache_read + cache_creation, output | promptTokenCount, candidatesTokenCount, cachedContentTokenCount | prompt_eval_count, eval_count |
| Streaming | SSE `data:` chunks, `[DONE]`, usage via `stream_options` | named SSE events (message_start … message_stop) | SSE with `?alt=sse`, each a partial response | NDJSON lines, final `done: true` |
| Blocked prompt | 400 / content_filter | `refusal` stop reason | no candidates + `promptFeedback.blockReason` | n/a |

Adapter decisions:
- Anthropic `temperature` is clamped to 0..1; consecutive same-role messages are merged.
- Anthropic `prompt_tokens` is reported OpenAI-style (including cache reads/writes) so cost math is uniform.
- Gemini/Ollama tool calls get generated ids (`call_<hex>`); Gemini tool results are mapped id→name.
- Gemini batch embeddings don't return usage, so the usage is estimated and flagged `estimated=true`.
- Google also offers a newer *Interactions* API with different streaming events. Not implemented here.
