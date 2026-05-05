# KT Experiment Web Manager

KT 实验框架的 Web 管理系统，用于启动、管理和监控训练任务。

## 快速启动

### 后端

```bash
# 安装依赖
cd web/backend
pip install -r requirements.txt

# 启动服务（默认端口 8765）
python -m uvicorn main:app --host 127.0.0.1 --port 8765
```

### 前端

```bash
# 安装依赖
cd web/frontend
npm install

# 启动开发服务器（默认端口 5173）
npm run dev
```

打开 http://localhost:5173 即可使用。

## 功能

- **环境管理** — 自动发现 pixi/conda 环境，支持自定义 Python 路径
- **动态参数表单** — 从现有模型注册表自动提取参数定义，渲染为可交互组件
- **任务管理** — 启动、停止、强制终止训练任务
- **实时日志** — 通过 WebSocket 实时推送训练输出（xterm.js 终端）
- **实验浏览** — 浏览 `runs/` 目录下已完成的实验记录和日志
- **GPU 监控** — 实时显示 GPU 使用率、显存、温度

## API 文档

后端启动后访问 http://localhost:8765/docs 查看 Swagger API 文档。

## 架构

```
web/
├── backend/          FastAPI + SQLite + WebSocket
│   ├── routers/      API 路由（6 组）
│   └── services/     业务逻辑（进程管理、环境发现、日志监听、GPU 监控）
└── frontend/         Vue 3 + TypeScript + Element Plus
    └── src/
        ├── api/          Axios API 封装
        ├── components/   可复用组件（动态表单、日志终端、布局）
        ├── composables/  Vue 组合式函数（WebSocket）
        └── views/        页面视图（5 个）
```

## 设计原则

- **零侵入** — 不修改 `train.py` 和 `utils/` 下的任何代码
- 仅通过 CLI 调用 + 只读导入注册表与现有框架交互
- Web 系统完全独立于训练框架，可随时移除
