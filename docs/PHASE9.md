# Phase 9 — 前端 Web 页面

> 把 Phase 8 的 REST API 包成一个**零依赖**的单文件 SPA，普通用户浏览器打开就能用。

## 1. 选型

| 方案 | 取舍 |
|---|---|
| ~~Vue 3 + Vite~~ | 太重，要 npm install / build |
| ~~Streamlit~~ | 不是"真"前端 |
| ✅ **纯 HTML + JS（单文件）** | 浏览器直接打开，零构建，零依赖 |

设计参考 ChatGPT：左侧栏（模式 + 会话列表），右侧聊天区。

## 2. 文件

- [web/index.html](../web/index.html) — 单文件 SPA（HTML + CSS + JS，共 ~21KB）
- [api/main.py](../api/main.py) — FastAPI 后端（Phase 8 已就绪）

## 3. 启动

```bash
# 1. 启后端
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. 打开前端
#    Windows: start web/index.html
#    或拖到浏览器
#    或右键 → 打开方式 → Chrome / Edge
```

> ⚠️ 直接打开 HTML 文件时，浏览器会从 `file://` 协议发起请求。CORS 已配置 `allow_origins=["*"]`，所以可以正常调 `http://localhost:8000`。

## 4. 界面

```
┌─────────────┬──────────────────────────────────┐
│ 🤖 AIAgent  │ ⚡ stream · 流式输出（SSE）       │
│ [+ 新对话]  │                              在线  │
│             │                                  │
│ 模式        │                                  │
│ 🔗 chain    │      ┌──────────────────────┐    │
│ 📚 rag      │      │ 💬                    │    │
│ 🤖 agent    │      │ 开始对话              │    │
│ 💬 multi    │      │ 选一个模式，输入问题   │    │
│ ⚡ stream    │      └──────────────────────┘    │
│             │                                  │
│ 会话        │                                  │
│ ☕ 咖啡机... │                                  │
│ 📦 退货...  ├──────────────────────────────────┤
│             │  [输入框________________] [↑]   │
└─────────────┴──────────────────────────────────┘
```

## 5. 5 种模式

| 模式 | 后端 endpoint | 前端体验 |
|---|---|---|
| `chain` | `POST /api/chat/chain` | 完整链路：清洗→分类→专家→校验 |
| `rag` | `POST /api/chat/rag` | 回答 + 显示 **📚 参考来源** 标签 |
| `agent` | `POST /api/chat/agent` | 回答 + 显示 **🛠️ 思考步骤** |
| `multi` | `POST /api/chat/multi` | 多轮记忆（同一 session_id） |
| `stream` | `POST /api/chat/stream` | SSE **打字机效果**（实时输出） |

## 6. 关键实现

### 6.1 会话管理（纯前端）
```js
state.history[sessionId] = [
  { role: "user", content: "..." },
  { role: "ai",   content: "...", sources: [...], tool_log: [...] },
];
```
- 切换会话 / 新建会话 / 删除会话（点击 `×`）
- session_id 存在 `localStorage`，刷新不丢

### 6.2 流式输出（SSE）
```js
const reader = r.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  // 解析 SSE 的 "data: ..." 行，append 到 aiMsg.content
  // 每次追加后 renderChat() → 看到打字机效果
}
```

### 6.3 RAG 来源展示
```js
// 后端返回的 metadata.sources
aiMsg.sources = data.metadata?.sources;
// 前端渲染：
m.sources.map(s => `<span class="source-tag">${s.source}</span>`)
// 显示成彩色标签条
```

### 6.4 Agent 思考日志
```js
aiMsg.tool_log = data.tool_log;
// 前端用 <pre> 等宽字体展示，方便读 ReAct 步骤
```

### 6.5 健康检查
启动后每 15s 调一次 `/api/health`，右上角显示"在线 · 109 chunks"。

## 7. 设计系统

| Token | 颜色 | 用途 |
|---|---|---|
| `--bg-primary` | `#0f1419` | 主背景（深色） |
| `--bg-secondary` | `#1a1f2e` | 侧边栏 / 顶栏 |
| `--bg-tertiary` | `#232938` | 输入框 / hover |
| `--accent` | `#10a37f` | OpenAI 绿（主色调） |
| `--text-secondary` | `#9ca3af` | 次要文字 |

完全用 CSS Variables 定义，**改主题色只需改 1 个值**。

## 8. 端到端验证

启动后端 + 打开 `web/index.html`，测试结果：

| 测试 | 结果 |
|---|---|
| `/api/health` 自动 ping | ✅ 顶部显示"在线 · 109 chunks" |
| RAG 模式问"咖啡机的水泵" | ✅ 回答"意大利ULKA，15Bar"，显示 5 个来源 |
| 多轮模式 | ✅ 轮 1 "好的，李四" → 轮 2 "李四"（记得名字） |
| Agent 模式问"现在几点" | ✅ 调 `get_current_time` 工具，输出时间 |
| Stream 模式 | ✅ 打字机效果，逐字显示 |
| 删除会话 | ✅ 侧边栏 `×` 按钮 + 后端 DELETE |
| 健康检查 15s 心跳 | ✅ 自动重连 |

## 9. 关键 Bug 修复

### Bug #4：LLM 端点慢，前端首次请求超时
**症状**：第一轮 RAG 请求偶尔超过 30s 超时。

**解决**：
- 前端：fetch 默认无超时，但 stream 模式自动逐 chunk 渲染
- 后端：保持默认（依赖 LangChain 客户端超时）
- 部署建议：生产加 nginx `proxy_read_timeout 300s`

## 10. 浏览器兼容

| 浏览器 | 支持 |
|---|---|
| Chrome / Edge 90+ | ✅ |
| Firefox 88+ | ✅ |
| Safari 14+ | ✅（SSE via fetch + ReadableStream） |
| IE 11 | ❌（用了 ReadableStream） |

## 11. 后续

- Phase 10：Docker 化（前端 + 后端 → 一个镜像）
- 加 Markdown 渲染（回答里的代码块、列表更漂亮）
- 加消息复制 / 重生成按钮
- 暗色 / 亮色主题切换
- PWA 化（离线可用）

## 12. 文件清单

| 文件 | 作用 |
|---|---|
| [web/index.html](../web/index.html) | 单文件 SPA（HTML + CSS + JS） |
| [api/main.py](../api/main.py) | FastAPI 后端（Phase 8） |
