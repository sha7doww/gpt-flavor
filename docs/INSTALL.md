# 安装 gpt-flavor SKILL

本 skill 遵循开放的 [AgentSkills](https://agentskills.io) 规范——目录就是 skill，复制到兼容 agent 的 skills 路径即可。不依赖 Python 包、不需要配 API key。

## 一眼看完

```bash
# 克隆本 repo
git clone https://github.com/sha7doww/gpt-flavor.git

# 拷到 Claude Code 全局 skills 路径
cp -r gpt-flavor/skill/gpt-flavor ~/.claude/skills/
```

Claude Code 里说 `/gpt-flavor` 或"用 GPT 味回答"即可触发。

## Claude Code

### 方式 A：全局（所有项目都能用）

```bash
mkdir -p ~/.claude/skills
cp -r skill/gpt-flavor ~/.claude/skills/gpt-flavor
```

或用 symlink（便于随 repo 更新同步）：

```bash
ln -s "$(pwd)/skill/gpt-flavor" ~/.claude/skills/gpt-flavor
```

### 方式 B：项目级（仅当前项目可用）

```bash
# 在目标项目的 git root 下
mkdir -p .claude/skills
cp -r /path/to/gpt-flavor/skill/gpt-flavor .claude/skills/
```

或 symlink：

```bash
ln -s /path/to/gpt-flavor/skill/gpt-flavor .claude/skills/gpt-flavor
```

### 验证

在 Claude Code 新开一个 session，输入：

```
/gpt-flavor
```

应能看到 skill 被激活并请求输入。或直接自然语言触发：

```
用 GPT 味写一段：我是不是太情绪化了？
```

对照 [SKILL.md 里的六招](../skill/gpt-flavor/SKILL.md#六招模板库) 检查输出是否覆盖"承接 → 反转 → 展开 → 金句 → 菜单 → 回问"。

## 其他 AgentSkills 兼容的 coding agent

因为 AgentSkills 是开放规范，下列 agent 均支持同一 `SKILL.md` 格式——只是 skills 路径不同：

| Agent | skills 路径 | 触发方式 |
|---|---|---|
| Claude Code | `~/.claude/skills/` 或 `.claude/skills/` | `/gpt-flavor` 或自然语言 |
| Cursor | 按 Cursor 文档配置 skills 目录 | 自然语言（Cursor 不一定支持斜杠命令） |
| Gemini CLI | `~/.gemini/skills/` | 自然语言 |
| OpenCode | 按 OpenCode 文档配置 | 自然语言 |
| Goose | 按 Goose 文档配置 | 自然语言 |

规范细节：<https://agentskills.io/specification>。兼容 agent 清单：<https://agentskills.io/home>。

## 卸载

```bash
rm -rf ~/.claude/skills/gpt-flavor          # 全局
rm -rf .claude/skills/gpt-flavor            # 项目级
```

symlink 同理，`rm` 安全。